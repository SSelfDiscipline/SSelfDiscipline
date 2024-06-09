@TryExcept(f"{PREFIX}ERROR")
def check_anchors(dataset, model, thr=4.0, imgsz=640):
    """
    用于train.py中,通过bpr确定是否需要改变anchors,需要就调用k-means重新计算anchors
    Evaluates anchor fit to dataset and adjusts if necessary, supporting customizable threshold and image size.
    :params dataset: 自定义数据集LoadImagesAndLabels返回的数据集
    :params model: 初始化的模型
    :params thr: 超参中得到  界定anchor与label匹配程度的阈值
    :params imgsz: 图片尺寸 默认640
   """
    # m: 从model中取出最后一层(Detect)
    m = model.module.model[-1] if hasattr(model, "module") else model.model[-1]  # Detect()
    # dataset.shapes.max(1, keepdims=True) = 每张图片的较长边
    # shapes: 将数据集图片的最长边缩放到img_size, 较小边相应缩放,得到新的所有数据集图片的宽高 [N, 2]  N训练集图片数量
    # imgaz:320, train训练集中有107张1366*768训练图, dataset.shapes:[[1366  768] ... [1366  768]], dataset.shapes.max(1, keepdims=True):[1366 ... 1366]
    # shapes:[[320, 179.91] ... [320, 179.91]]
    shapes = imgsz * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    # 产生随机数scale (107, 1)
    scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))  # augment scale
    # torch.Size([855, 2])  所有target(855个)基于原图大小的wh     shapes * scale: 随机化尺度变化,锚框宽高乘640并随机缩放
    wh = torch.tensor(np.concatenate([l[:, 3:5] * s for s, l in zip(shapes * scale, dataset.labels)])).float()  # wh
    # tensor([[15.48895, 15.88619],
    #         [16.28319, 20.65190],
    #         [20.25471, 14.29750],
    #         ...,
    #         [15.86208, 10.57459],
    #         [15.38139, 13.93932],
    #         [18.26520, 19.22661]])
    def metric(k):
        """用在check_anchors函数中  compute metric
        根据数据集的所有图片的wh和当前所有anchors k计算 bpr(best possible recall) 和 aat(anchors above threshold)
        :params k: anchors [9, 2]  wh: [N, 2]
        :return bpr: best possible recall 最多能被召回(通过thr)的gt框数量 / 所有gt框数量   小于0.98 才会用k-means计算anchor
        :return aat: anchors above threshold 每个target平均有多少个anchors
        """
        # None添加维度  所有target(gt)的wh wh[:, None] [855, 2]->[855, 1, 2]
        #             所有anchor的wh k[None] [9, 2]->[1, 9, 2]
        # r: target的高h宽w与anchor的高h_a宽w_a的比值，即h/h_a, w/w_a  [855, 9, 2]  有可能大于1，也可能小于等于1
        r = wh[:, None] / k[None]
        # x 高宽比和宽高比的最小值 无论r大于1，还是小于等于1最后统一结果都要小于1   [855, 9]
        x = torch.min(r, 1 / r).min(2)[0]  # ratio metric
        # best [855] 为每个gt框选择匹配所有anchors宽高比例值最好的那一个比值
        best = x.max(1)[0]  # best_x
        # aat(anchors above threshold)  [1]     每个target平均有多少个anchors
        # sum(axis), # 当axis=1时，求的是每一行元素的和
        aat = (x > 1 / thr).float().sum(1).mean()  # anchors above threshold
        # bpr(best possible recall) = 最多能被召回(通过thr)的gt框数量 / 所有gt框数量   小于0.98 才会用k-means计算anchor
        # 所有标签与anchors宽高比例列表x中取每行最大的比值得到best列表，best中大于1/阈值的比值的平均值
        bpr = (best > 1 / thr).float().mean()  # best possible recall
        return bpr, aat
 
    # stride:tensor([[[ 8.]],
    # 
    #               [[16.]],
    # 
    #               [[32.]]])
    stride = m.stride.to(m.anchors.device).view(-1, 1, 1)  # model strides
 
    # print(m.anchors)
    # tensor([[[ 1.25000,  1.62500],
    #          [ 2.00000,  3.75000],
    #          [ 4.12500,  2.87500]],
    # 
    #         [[ 1.87500,  3.81250],
    #          [ 3.87500,  2.81250],
    #          [ 3.68750,  7.43750]],
    # 
    #         [[ 3.62500,  2.81250],
    #          [ 4.87500,  6.18750],
    #          [11.65625, 10.18750]]])
    # anchors: [N,2]  所有anchors的宽高   基于缩放后的图片大小(较长边为640 较小边相应缩放)
    anchors = m.anchors.clone() * stride  # current anchors
    # print(anchors)
    # tensor([[[ 10.,  13.],
    #          [ 16.,  30.],
    #          [ 33.,  23.]],
    # 
    #         [[ 30.,  61.],
    #          [ 62.,  45.],
    #          [ 59., 119.]],
    # 
    #         [[116.,  90.],
    #          [156., 198.],
    #          [373., 326.]]])
    
    # 计算出数据集所有图片的wh和当前所有anchors的bpr和aat
    # bpr: bpr(best possible recall): 最多能被召回(通过thr)的gt框数量 / 所有gt框数量  [1] 0.96223  小于0.98 才会用k-means计算anchor
    # aat(anchors past thr): [1] 3.54360 通过阈值的anchor个数
    bpr, aat = metric(anchors.cpu().view(-1, 2))
    s = f"\n{PREFIX}{aat:.2f} anchors/target, {bpr:.3f} Best Possible Recall (BPR). "
    # threshold to recompute
    # 考虑这9类anchor的宽高和gt框的宽高之间的差距, 如果bpr<0.98(说明当前anchor不能很好的匹配数据集gt框)就会根据k-means算法重新聚类新的anchor
    if bpr > 0.98:  # threshold to recompute
        LOGGER.info(f"{s}Current anchors are a good fit to dataset ✅")
    else:
        LOGGER.info(f"{s}Anchors are a poor fit to dataset ⚠️, attempting to improve...")
        na = m.anchors.numel() // 2  # number of anchors
        # 如果bpr<0.98(最大为1 越大越好) 使用k-means + 遗传进化算法选择出与数据集更匹配的anchors框  [9, 2]
        anchors = kmean_anchors(dataset, n=na, img_size=imgsz, thr=thr, gen=1000, verbose=False)
        # 计算新的anchors的new_bpr
        new_bpr = metric(anchors)[0]
        # 比较k-means + 遗传进化算法进化后的anchors的new_bpr和原始anchors的bpr
        # 注意: 这里并不一定进化后的bpr必大于原始anchors的bpr, 因为两者的衡量标注是不一样的  进化算法的衡量标准是适应度 而这里比的是bpr
        if new_bpr > bpr:  # replace anchors
            anchors = torch.tensor(anchors, device=m.anchors.device).type_as(m.anchors)
            # 替换m的anchors
            m.anchors[:] = anchors.clone().view_as(m.anchors)
            # 检查anchor顺序和stride顺序是否一致 不一致就调整
            # 因为我们的m.anchors是相对各个feature map 所以必须要顺序一致 否则效果会很不好
            check_anchor_order(m)  # must be in pixel-space (not grid-space)
            # 替换m的anchors(相对各个feature map)      [9, 2] -> [3, 3, 2]
            m.anchors /= stride
            s = f"{PREFIX}Done ✅ (optional: update model *.yaml to use these anchors in the future)"
        else:
            s = f"{PREFIX}Done ⚠️ (original anchors better than new anchors, proceeding with original anchors)"
        LOGGER.info(s)