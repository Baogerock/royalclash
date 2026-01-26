import cv2

# 回调函数：处理鼠标事件
def click_event(event, x, y, flags, params):
    # 实时更新图像，为了显示动态的坐标数值
    if event == cv2.EVENT_MOUSEMOVE:
        img_copy = img.copy()
        text = f'Coords: ({x}, {y})'
        cv2.putText(img_copy, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        cv2.imshow('Image', img_copy)

    # 左键点击时打印坐标
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked Pixel Position: x={x}, y={y}")
        # 在点击位置画一个红色小圆点（可选）
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('Image', img)

# 1. 加载图片 (请确保替换为你本地图片的路径)
image_path = 'a.png'
img = cv2.imread(image_path)

if img is None:
    print("错误：无法加载图片，请检查路径。")
else:
    # 2. 创建窗口并绑定鼠标回调函数
    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', click_event)

    print("操作提示：")
    print("- 移动鼠标查看坐标")
    print("- 点击左键打印并标记坐标")
    print("- 按 'q' 或 'Esc' 退出程序")

    # 3. 初始显示
    cv2.imshow('Image', img)

    # 4. 等待按键退出
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()