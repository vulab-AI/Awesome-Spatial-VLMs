from PIL import Image, ImageDraw, ImageFont

def draw_dot_matrix_2d(img: Image.Image, dots_size_w=6, dots_size_h=6, font_path="fonts/arial.ttf") -> Image.Image:
    """
    在 PIL 图片上绘制二维点阵坐标 (x,y)，返回绘制后的 PIL 图片。
    """
    # 确保 RGB 模式
    if img.mode != "RGB":
        img = img.convert("RGB")
    draw = ImageDraw.Draw(img, "RGB")

    width, height = img.size
    grid_size_w = dots_size_w + 1
    grid_size_h = dots_size_h + 1
    cell_width = width / grid_size_w
    cell_height = height / grid_size_h

    # 设置字体
    font = ImageFont.truetype(font_path, width // 40)

    count = 0
    for j in range(1, grid_size_h):
        for i in range(1, grid_size_w):
            x = int(i * cell_width)
            y = int(j * cell_height)

            pixel_color = img.getpixel((x, y))
            opposite_color = (0, 0, 0) if sum(pixel_color) >= 255 * 3 / 2 else (255, 255, 255)

            circle_radius = max(2, width // 240)
            draw.ellipse(
                [(x - circle_radius, y - circle_radius), (x + circle_radius, y + circle_radius)],
                fill=opposite_color
            )

            count_w = count // dots_size_w
            count_h = count % dots_size_w
            label_str = f"({count_w+1},{count_h+1})"
            draw.text((x + 3, y), label_str, fill=opposite_color, font=font)

            count += 1

    return img


def draw_dot_matrix_3d(imgs: list[Image.Image], dots_size_w=6, dots_size_h=6, font_path="fonts/arial.ttf") -> list[Image.Image]:
    """
    在 PIL 图片序列上绘制三维点阵坐标 (t,x,y)，返回绘制后的 PIL 图片列表。
    """
    processed_imgs = []
    for t, img in enumerate(imgs, start=1):
        if img.mode != "RGB":
            img = img.convert("RGB")
        draw = ImageDraw.Draw(img, "RGB")

        width, height = img.size
        grid_size_w = dots_size_w + 1
        grid_size_h = dots_size_h + 1
        cell_width = width / grid_size_w
        cell_height = height / grid_size_h

        font = ImageFont.truetype(font_path, width // 40)

        count = 0
        for j in range(1, grid_size_h):
            for i in range(1, grid_size_w):
                x = int(i * cell_width)
                y = int(j * cell_height)

                pixel_color = img.getpixel((x, y))
                opposite_color = (0, 0, 0) if sum(pixel_color) >= 255 * 3 / 2 else (255, 255, 255)

                circle_radius = max(2, width // 240)
                draw.ellipse(
                    [(x - circle_radius, y - circle_radius), (x + circle_radius, y + circle_radius)],
                    fill=opposite_color
                )

                count_w = count // dots_size_w
                count_h = count % dots_size_w
                label_str = f"({t},{count_w+1},{count_h+1})"
                draw.text((x + 3, y), label_str, fill=opposite_color, font=font)

                count += 1

        processed_imgs.append(img)

    return processed_imgs


def process(imgs):
    #if imgs>1:
    if len(imgs)>1:
        imgs = draw_dot_matrix_3d(imgs, dots_size_w=6, dots_size_h=6, font_path="fonts/arial.ttf")
    else:
        imgs[0] = draw_dot_matrix_2d(imgs[0], dots_size_w=6, dots_size_h=6, font_path="fonts/arial.ttf")
    return imgs