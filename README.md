# MagiCatzOCR
一个识别音乐游戏Deemo当中MagiCatz结算图的程序，原理是kNN。对于截屏和照片都有很好的识别效果，屏幕部分遮挡（如上隐板等）也不影响识别效果。

包含三个py程序，第1个用来截取_photos中各图片中的物量和FC/AC图标，第二个用来产生训练集数据，第三个用来对输入的结算图进行识别识别。依次运行三个程序即可得出结果，对第三个程序进行一些修改即可实装到QQ机器人上。

_photos中，图片1xx、2xx、3xx分别为Easy难度AC、Normal难度FC、Hard难度未FC结算图的照片。其中x01-x10是新版本的结算图，x11-x15为旧版本做了适配。

你也可以通过更改_image和_photos下的图片，来识别其他曲目的结算图。

效果如下图所示。*（其中AC图截取于[FlowerColor的手元](https://www.bilibili.com/video/av9557280)）*

 ![Alt text](https://github.com/Excaive/MagiCatzOCR/blob/master/imgREADME1.jpg)
