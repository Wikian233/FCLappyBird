from PIL import Image

# 打开图片
img = Image.open('img/background.png')

# 创建一个同样大小的黑色图片
black_img = Image.new('RGB', img.size, color='black')

# 保存黑色图片
black_img.save('path_to_save_black_image.png')
