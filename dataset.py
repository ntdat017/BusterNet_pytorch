import os
import numpy as np

import cv2
import json
import lmdb
import pyarrow as pa

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class USCISIDataset(Dataset):
    def __init__(self, lmdb_dir, sample_file, transform=None, target_transform=None, differentiate_target=True):
        super(USCISIDataset, self).__init__()
        
        self.lmdb_dir = lmdb_dir
        self.sample_file = os.path.join(lmdb_dir, sample_file)
        self.transform = transform
        self.target_transform = target_transform
        self.differentiate_target = differentiate_target

        assert os.path.isdir(lmdb_dir)
        self.lmdb_dir = lmdb_dir
        assert os.path.isfile(self.sample_file)

        self._init_db()


    def _init_db(self):
        self.env = lmdb.open(self.lmdb_dir, subdir=os.path.isdir(self.lmdb_dir),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        # with self.env.begin(write=False) as txn:
        #     self.length = txn.stat()['entries'] - 1
        self.keys = self._load_sample_keys()
    
    def _load_sample_keys(self) :
        with open(self.sample_file, 'r') as f:
            keys = [line.strip() for line in f.readlines()]
        return keys

    def __getitem__(self, index: int):
        img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            lut_str = txn.get(self.keys[index].encode())
            img, cmd_mask, trans_mat = self._decode_lut_str(lut_str)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            cmd_mask = cmd_mask.astype(np.uint8)
            cmd_mask = self.target_transform(cmd_mask)
            cmd_mask = (cmd_mask * 256).type(torch.int32)

        return img, cmd_mask, trans_mat

    def __len__(self) -> int:
        return len(self.keys)


    def _decode_lut_str(self, lut_str) :
        '''Decode a raw LMDB lut
        INPUT:
            lut_str = str, raw string retrieved from LMDB
        OUTPUT: 
            image = np.ndarray, dtype='uint8', cmd image
            cmd_mask = np.ndarray, dtype='float32', cmd mask
            trans_mat = np.ndarray, dtype='float32', cmd transform matrix
        '''
        # 1. get raw lut
        lut = json.loads(lut_str)
        # 2. reconstruct image
        image = self._get_image_from_lut(lut)
        # 3. reconstruct copy-move masks
        cmd_mask = self._get_mask_from_lut(lut)
        # 4. get transform matrix if necessary
        trans_mat = self._get_transmat_from_lut(lut)
        return ( image, cmd_mask, trans_mat )

        
    def _get_image_from_lut( self, lut ) :
        '''Decode image array from LMDB lut
        INPUT:
            lut = dict, raw decoded lut retrieved from LMDB
        OUTPUT:
            image = np.ndarray, dtype='uint8'
        '''
        image_jpeg_buffer = lut['image_jpeg_buffer']
        image = cv2.imdecode( np.array(image_jpeg_buffer).astype('uint8').reshape([-1,1]), 1 )
        return image

    def _get_mask_from_lut( self, lut ) :
        '''Decode copy-move mask from LMDB lut
        INPUT:
            lut = dict, raw decoded lut retrieved from LMDB
        OUTPUT:
            cmd_mask = np.ndarray, dtype='float32'
                       shape of HxWx1, if differentiate_target=False
                       shape of HxWx3, if differentiate target=True
        NOTE:
            cmd_mask is encoded in the one-hot style, if differentiate target=True.
            color channel, R, G, and B stand for TARGET, SOURCE, and BACKGROUND classes
        '''
        def reconstruct( cnts, h, w, val=1 ) :
            rst = np.zeros([h,w], dtype='uint8')
            cv2.fillPoly( rst, cnts, val )
            return rst 
        h, w = lut['image_height'], lut['image_width']
        src_cnts = [ np.array(cnts).reshape([-1,1,2]) for cnts in lut['source_contour'] ]
        src_mask = reconstruct( src_cnts, h, w, val = 1 )
        tgt_cnts = [ np.array(cnts).reshape([-1,1,2]) for cnts in lut['target_contour'] ]
        tgt_mask = reconstruct( tgt_cnts, h, w, val = 1 )
        if ( self.differentiate_target ) :
            # 3-class target
            background = np.ones([h,w]).astype('uint8') - np.maximum(src_mask, tgt_mask)
            cmd_mask = np.dstack( [tgt_mask, src_mask, background ] ).astype(np.float32)
        else :
            # 2-class target
            cmd_mask = np.maximum(src_mask, tgt_mask).astype(np.float32)
        return cmd_mask
    def _get_transmat_from_lut( self, lut ) :
        '''Decode transform matrix between SOURCE and TARGET
        INPUT:
            lut = dict, raw decoded lut retrieved from LMDB
        OUTPUT:
            trans_mat = np.ndarray, dtype='float32', size of 3x3
        '''
        trans_mat = lut['transform_matrix']
        return np.array(trans_mat).reshape([3,3])



# def collater(data):
#     imgs = [s['img'] for s in data]
#     annots = [s['annot'] for s in data]
#     scales = [s['scale'] for s in data]

#     imgs = torch.from_numpy(np.stack(imgs, axis=0))

#     max_num_annots = max(annot.shape[0] for annot in annots)

#     if max_num_annots > 0:

#         annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

#         for idx, annot in enumerate(annots):
#             if annot.shape[0] > 0:
#                 annot_padded[idx, :annot.shape[0], :] = annot
#     else:
#         annot_padded = torch.ones((len(annots), 1, 5)) * -1

#     imgs = imgs.permute(0, 3, 1, 2)

#     return {'img': imgs, 'annot': annot_padded, 'scale': scales}