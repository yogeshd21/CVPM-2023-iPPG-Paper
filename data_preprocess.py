## Author: Yogesh Deshpande Aug 2021 - May 2023

from preprocess import preprocesses

input_datadir = './Frames'
output_datadir = './aligned_img'

obj=preprocesses(input_datadir,output_datadir)
nrof_images_total,nrof_successfully_aligned=obj.collect_data()

print('Total number of images: %d' % nrof_images_total)
print('Number of successfully aligned images: %d' % nrof_successfully_aligned)