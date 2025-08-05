import cv2 
import numpy as np 
import pickle 
import argparse
import os 
from time import time

from utils import * 

def FeatMatch(opts, data_files=[]): 
    print(f"Starting FeatMatch with data_dir: {opts.data_dir}")
    print(f"Output dir: {opts.out_dir}")
    print(f"Allowed extensions: {opts.ext}")
    
    if len(data_files) == 0: 
        print(f"Scanning directory: {opts.data_dir}")
        all_files = sorted(os.listdir(opts.data_dir))
        print(f"Found files: {all_files}")
        print(f"Extension check details:")
        for x in all_files:
            ext = x.split('.')[-1].lower()
            match = ext in opts.ext
            print(f"  {x} -> extension '{ext}' -> match: {match}")
        
        # Filter files and create corresponding paths and names
        filtered_files = [x for x in all_files if x.split('.')[-1].lower() in opts.ext]
        img_paths = [os.path.join(opts.data_dir, x) for x in filtered_files]
        img_names = filtered_files  # Use the same filtered list
        
        print(f"Filtered image paths: {img_paths}")
        print(f"Corresponding image names: {img_names}")
    
    else: 
        img_paths = data_files
        img_names = sorted([x.split('/')[-1] for x in data_files])
        print(f"Using provided data files: {img_paths}")
        
    feat_out_dir = os.path.join(opts.out_dir,'features',opts.features)
    matches_out_dir = os.path.join(opts.out_dir,'matches',opts.matcher)

    print(f"Creating output directories:")
    print(f"  Features: {feat_out_dir}")
    print(f"  Matches: {matches_out_dir}")
    
    if not os.path.exists(feat_out_dir): 
        os.makedirs(feat_out_dir)
    if not os.path.exists(matches_out_dir): 
        os.makedirs(matches_out_dir)
    
    data = []
    t1 = time()
    if len(img_paths) == 0:
        print("ERROR: No images found! Check your data directory and file extensions.")
        return
        
    print(f"Processing {len(img_paths)} images...")
    
    for i, img_path in enumerate(img_paths): 
        print(f"Processing image {i+1}/{len(img_paths)}: {img_path}")
        img = cv2.imread(img_path)
        if img is None:
            print(f"WARNING: Could not read image {img_path}")
            continue
            
        img_name = img_names[i].split('.')[0]
        print(f"  Processing as image name: {img_name}")
        img = img[:,:,::-1]

        # Try to use xfeatures2d first, fallback to standard features
        try:
            feat = getattr(cv2.xfeatures2d, '{}_create'.format(opts.features))()
        except AttributeError:
            if opts.features == 'SURF':
                print(f"SURF not available, using SIFT instead")
                feat = cv2.SIFT_create()
            elif opts.features == 'SIFT':
                feat = cv2.SIFT_create()
            else:
                print(f"Feature {opts.features} not available, using ORB instead")
                feat = cv2.ORB_create()
        kp, desc = feat.detectAndCompute(img,None)
        print(f"  Found {len(kp)} keypoints")
        data.append((img_name, kp, desc))

        kp_ = SerializeKeypoints(kp)
        
        kp_path = os.path.join(feat_out_dir, 'kp_{}.pkl'.format(img_name))
        desc_path = os.path.join(feat_out_dir, 'desc_{}.pkl'.format(img_name))
        
        print(f"  Saving keypoints to: {kp_path}")
        with open(kp_path,'wb') as out:
            pickle.dump(kp_, out)

        print(f"  Saving descriptors to: {desc_path}")
        with open(desc_path,'wb') as out:
            pickle.dump(desc, out)
            
        # Verify files were created
        if os.path.exists(kp_path) and os.path.exists(desc_path):
            print(f"  ✓ Files successfully saved for {img_name}")
        else:
            print(f"  ✗ ERROR: Failed to save files for {img_name}")

        if opts.save_results: 
            # Create visualization directory
            vis_out_dir = os.path.join(opts.out_dir, 'visualizations')
            if not os.path.exists(vis_out_dir):
                os.makedirs(vis_out_dir)
            
            # Draw simple red dots for keypoints
            img_with_kp = img[:,:,::-1].copy()
            for keypoint in kp:
                x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
                cv2.circle(img_with_kp, (x, y), 3, (0, 0, 255), -1)  # Red filled circles
            
            # Save visualization
            vis_path = os.path.join(vis_out_dir, f'{img_name}_keypoints.png')
            cv2.imwrite(vis_path, img_with_kp)
            print(f"  Saved keypoint visualization: {vis_path}")

        t2 = time()

        if (i % opts.print_every) == 0:    
            print(f"FEATURES DONE: {i+1}/{len(img_paths)} [time={(t2-t1):.2f}s]")

        t1 = time()

    num_done = 0 
    num_matches = int(((len(data)-1) * (len(data))) / 2)
    print(f"\nFeature extraction completed. Data contains {len(data)} images:")
    for img_name, kp, desc in data:
        print(f"  {img_name}: {len(kp)} keypoints")
    
    print(f"\nStarting feature matching for {len(data)} images ({num_matches} pairs)...")

    t1 = time()
    for i in range(len(data)): 
        for j in range(i+1, len(data)): 
            img_name1, kp1, desc1 = data[i]
            img_name2, kp2, desc2 = data[j]

            matcher = getattr(cv2,opts.matcher)(crossCheck=opts.cross_check)
            matches = matcher.match(desc1,desc2)

            matches = sorted(matches, key = lambda x:x.distance)
            matches_ = SerializeMatches(matches)

            pickle_path = os.path.join(matches_out_dir, 'match_{}_{}.pkl'.format(img_name1,
                                                                                 img_name2))
            with open(pickle_path,'wb') as out:
                pickle.dump(matches_, out)

            print(f"  {img_name1} <-> {img_name2}: {len(matches)} matches")
            num_done += 1 
            t2 = time()

            if (num_done % opts.print_every) == 0: 
                print(f"MATCHES DONE: {num_done}/{num_matches} [time={(t2-t1):.2f}s]")

            t1 = time()
    
    # Report total matches
    total_matches = sum([len(SerializeMatches(matcher.match(data[i][2], data[j][2]))) 
                        for i in range(len(data)) for j in range(i+1, len(data))])
    print(f"\nTotal matches found: {total_matches}")
    print("Feature matching completed.")
            


def SetArguments(parser): 

    #directories stuff
    parser.add_argument('--data-files',action='store',type=str,default='',dest='data_files') 
    parser.add_argument('--data-dir',action='store',type=str,default='../data/fountain-P11/images/',
                        dest='data_dir',help='directory containing images (default: ../data/\
                        fountain-P11/images/)') 
    parser.add_argument('--ext',action='store',type=str,default='jpg,png',dest='ext',
                        help='comma seperated string of allowed image extensions \
                        (default: jpg,png)') 
    parser.add_argument('--out-dir',action='store',type=str,default='',
                        dest='out_dir',help='root directory to store results in \
                        (default: same as data-dir parent)') 

    #feature matching args
    parser.add_argument('--features',action='store', type=str, default='SIFT', dest='features',
                        help='[SIFT|ORB] Feature algorithm to use (default: SIFT)') 
    parser.add_argument('--matcher',action='store',type=str,default='BFMatcher',dest='matcher',
                        help='[BFMatcher|FlannBasedMatcher] Matching algorithm to use \
                        (default: BFMatcher)') 
    parser.add_argument('--cross-check',action='store_true',default=False,dest='cross_check',
                        help='Whether to cross check feature matching or not \
                        (default: False)') 
    
    #misc
    parser.add_argument('--print-every',action='store', type=int, default=1, dest='print_every',
                        help='[1,+inf] print progress every print_every seconds, -1 to disable \
                        (default: 1)')
    parser.add_argument('--save-results',action='store_true',default=False, 
                        dest='save_results',help='whether to save images with\
                        keypoints drawn on them (default: False)')  

def PostprocessArgs(opts): 
    opts.ext = [x for x in opts.ext.split(',')]
    
    opts.data_files_ = []
    if opts.data_files != '': 
        opts.data_files_ = opts.data_files.split(',')
    opts.data_files = opts.data_files_
    
    # Set default output directory if not specified
    if opts.out_dir == '':
        opts.out_dir = os.path.dirname(opts.data_dir) if opts.data_dir.endswith('/') else os.path.dirname(opts.data_dir + '/')
        if opts.out_dir == '':
            opts.out_dir = '.'

if __name__=='__main__': 
    parser = argparse.ArgumentParser()
    SetArguments(parser)
    opts = parser.parse_args()
    PostprocessArgs(opts)

    FeatMatch(opts, opts.data_files)