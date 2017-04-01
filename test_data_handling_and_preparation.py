import unittest

import numpy as np
import data_handling_and_preparation as dhap

class TestDataHandlingAndPreparation(unittest.TestCase):


    def test_get_new_dimensions(self):

        h = 3744
        w = 5616
        patch_h = 400
        patch_w = 400

        nh, nw, nh_slices, nw_slices = dhap.get_new_dimensions(h, w, patch_h, patch_w)

        self.assertEqual(nh, 3600)
        self.assertEqual(nw, 5600)
        self.assertEqual(nh_slices, 9)
        self.assertEqual(nw_slices, 14)


    def test_slice_the_image_into_patches(self):
        
        image = np.random.randint(0, 255 + 1, size=(10, 15, 3))
        
        patches_list = dhap.slice_the_image_into_patches(image,
                                                         patch_h = 5,
                                                         patch_w = 5)

        condition_1 = np.array_equal(image[0:5,   0:5, :], patches_list[0][0])
        condition_2 = np.array_equal(image[0:5,  5:10, :], patches_list[0][1])
        condition_3 = np.array_equal(image[0:5, 10:15, :], patches_list[0][2])

        condition_4 = np.array_equal(image[5:10,   0:5, :], patches_list[1][0])
        condition_5 = np.array_equal(image[5:10,  5:10, :], patches_list[1][1])
        condition_6 = np.array_equal(image[5:10, 10:15, :], patches_list[1][2])

        self.assertTrue(condition_1)
        self.assertTrue(condition_2)
        self.assertTrue(condition_3)

        self.assertTrue(condition_4)
        self.assertTrue(condition_5)
        self.assertTrue(condition_6)


    def test_slice_the_mask_into_patches(self):
        
        mask = np.random.randint(0, 255 + 1, size=(10, 15))
        
        patches_list = dhap.slice_the_mask_into_patches(mask,
                                                        patch_h = 5,
                                                        patch_w = 5)

        condition_1 = np.array_equal(mask[0:5,   0:5], patches_list[0][0])
        condition_2 = np.array_equal(mask[0:5,  5:10], patches_list[0][1])
        condition_3 = np.array_equal(mask[0:5, 10:15], patches_list[0][2])

        condition_4 = np.array_equal(mask[5:10,   0:5], patches_list[1][0])
        condition_5 = np.array_equal(mask[5:10,  5:10], patches_list[1][1])
        condition_6 = np.array_equal(mask[5:10, 10:15], patches_list[1][2])

        self.assertTrue(condition_1)
        self.assertTrue(condition_2)
        self.assertTrue(condition_3)

        self.assertTrue(condition_4)
        self.assertTrue(condition_5)
        self.assertTrue(condition_6)


    def test_combine_pathes_into_image(self):

        image = np.random.randint(0, 255 + 1, size=(10, 15, 3))
        
        patches_list = [[np.zeros((5, 5, 3)) for j in range(3)] for i in range(2)]
        patches_list[0][0] = image[0:5,   0:5, :]
        patches_list[0][1] = image[0:5,  5:10, :]
        patches_list[0][2] = image[0:5, 10:15, :]

        patches_list[1][0] = image[5:10,   0:5, :]
        patches_list[1][1] = image[5:10,  5:10, :]
        patches_list[1][2] = image[5:10, 10:15, :]

        combined_image = dhap.combine_pathes_into_image(patches_list)
        condition = np.array_equal(image, combined_image)

        self.assertTrue(condition)


    def test_combine_pathes_into_mask(self):

        mask = np.random.randint(0, 255 + 1, size=(10, 15))
        
        patches_list = [[np.zeros((5, 5)) for j in range(3)] for i in range(2)]
        patches_list[0][0] = mask[0:5,   0:5]
        patches_list[0][1] = mask[0:5,  5:10]
        patches_list[0][2] = mask[0:5, 10:15]

        patches_list[1][0] = mask[5:10,   0:5]
        patches_list[1][1] = mask[5:10,  5:10]
        patches_list[1][2] = mask[5:10, 10:15]

        combined_mask = dhap.combine_pathes_into_mask(patches_list)
        condition = np.array_equal(mask, combined_mask)

        self.assertTrue(condition)



if __name__ == '__main__':
    unittest.main()






