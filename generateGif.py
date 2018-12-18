import imageio

images = []
for e in range(49):
    img_name = 'images/' + str(e + 1) + '.png'
    images.append(imageio.imread(img_name))
imageio.mimsave('images/generation_animation.gif', images, fps=5)
print('Export gif to images/generation_animation.gif')
