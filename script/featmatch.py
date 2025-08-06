import os
import cv2
import numpy as np
from pickle import load, dump
from argparse import ArgumentParser
from tqdm import tqdm

from utils import * 


def FeatMatch(opts): 
    print(f"Starting FeatMatch with data_dir: {opts.data_dir}")
    print(f"Output dir: {opts.out_dir}")
    print(f"Allowed extensions: {opts.ext}")
    
    img_paths = [os.path.join(opts.data_dir, x) for x in sorted(os.listdir(opts.data_dir)) if x.split('.')[-1].lower() in opts.ext]
    img_names = [x.split('.')[0] for x in sorted(os.listdir(opts.data_dir)) if x.split('.')[-1].lower() in opts.ext]
    print(f"Processing {len(img_paths)} images...")

    feat_out_dir = os.path.join(opts.out_dir, 'features', opts.features)
    matches_out_dir = os.path.join(opts.out_dir, 'matches', opts.matcher)
    
    os.makedirs(feat_out_dir, exist_ok=True)
    os.makedirs(matches_out_dir, exist_ok=True)

    if opts.save_matches_vis:
        vis_matches_out_dir = os.path.join(opts.out_dir, 'matches_vis', opts.features)
        os.makedirs(vis_matches_out_dir, exist_ok=True)

    # --- Feature Extraction ---
    data = []
    print("\nExtracting features...")
    for i, img_path in tqdm(enumerate(img_paths), total=len(img_paths), desc="Extracting features"):
        img_name = img_names[i]
        
        # Check if features already exist
        kp_path = os.path.join(feat_out_dir, f'kp_{img_name}.pkl')
        desc_path = os.path.join(feat_out_dir, f'desc_{img_name}.pkl')

        if os.path.exists(kp_path) and os.path.exists(desc_path):
             with open(kp_path, 'rb') as f: kp = DeserializeKeypoints(load(f))
             with open(desc_path, 'rb') as f: desc = load(f)
        else:
            img = cv2.imread(img_path)
            if img is None: continue
            
            feat = cv2.SIFT_create(nfeatures=opts.max_features)
            kp, desc = feat.detectAndCompute(img, None)

            with open(kp_path, 'wb') as f:
                dump(SerializeKeypoints(kp), f)
            with open(desc_path, 'wb') as f:
                dump(desc, f)
        
        data.append((img_name, kp, desc))
    print("Feature extraction complete.")

    # --- Feature Matching ---
    print(f"\nStarting feature matching with geometric verification...")
    
    total_pairs = sum(range(len(data)))
    with tqdm(total=total_pairs, desc="Matching image pairs") as pbar:
        for i in range(len(data)):
            for j in range(i + 1, len(data)):
                pbar.update(1)
                img_name1, kp1, desc1 = data[i]
                img_name2, kp2, desc2 = data[j]

                # 1. KNN matching with Lowe's Ratio Test
                matcher = getattr(cv2, opts.matcher)()
                knn_matches = matcher.knnMatch(desc1, desc2, k=2)
                
                ratio_matches = []
                for match_pair in knn_matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < opts.ratio_threshold * n.distance:
                            ratio_matches.append(m)
                
                # --- GEOMETRIC VERIFICATION STEP ---
                good_matches = []
                # At least 8 points are required to estimate the Fundamental Matrix
                if len(ratio_matches) > 8:
                    # Get the coordinates of the matched keypoints
                    pts1 = np.float32([kp1[m.queryIdx].pt for m in ratio_matches])
                    pts2 = np.float32([kp2[m.trainIdx].pt for m in ratio_matches])

                    # Estimate Fundamental Matrix using RANSAC
                    F, mask = cv2.findFundamentalMat(pts1, pts2,
                                                    method=opts.fund_method,
                                                    ransacReprojThreshold=opts.outlier_thres,
                                                    confidence=opts.fund_prob)
                    
                    # Use the mask to select only inlier matches
                    if mask is not None:
                        good_matches = np.array(ratio_matches)[mask.ravel() == 1].tolist()

                if len(good_matches) < opts.min_matches:
                    # print(f"  {img_name1} <-> {img_name2}: {len(good_matches)} inliers (SKIPPED - too few)")
                    good_matches = [] # Save empty list
                # else:
                #     print(f"  {img_name1} <-> {img_name2}: {len(good_matches)} inliers (from {len(ratio_matches)} ratio-test matches)")

                # Save visualization and matches using the *geometrically verified* good_matches
                if opts.save_matches_vis and len(good_matches) > 0:
                    img1 = cv2.imread(img_paths[i])
                    img2 = cv2.imread(img_paths[j])
                    # Draw top 50 matches sorted by distance
                    drawn_matches = sorted(good_matches, key=lambda x: x.distance)[:50]
                    match_vis = cv2.drawMatches(img1, kp1, img2, kp2, drawn_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                    vis_path = os.path.join(vis_matches_out_dir, f'match_{img_name1}_{img_name2}.jpg')
                    cv2.imwrite(vis_path, match_vis)

                with open(os.path.join(matches_out_dir, f'match_{img_name1}_{img_name2}.pkl'), 'wb') as f:
                    dump(SerializeMatches(good_matches), f)
    
    total_matches = 0
    valid_pairs = 0
    for i in range(len(data)):
        for j in range(i+1, len(data)):
            img_name1, _, _ = data[i]
            img_name2, _, _ = data[j]
            pickle_path = os.path.join(matches_out_dir, f'match_{img_name1}_{img_name2}.pkl')
            if os.path.exists(pickle_path):
                with open(pickle_path, 'rb') as f:
                    matches = load(f)
                    if len(matches) > 0:
                        total_matches += len(matches)
                        valid_pairs += 1
    
    print(f"\nTotal inlier matches found: {total_matches} across {valid_pairs} valid pairs")
    print("Feature matching completed.")


def set_arguments(parser):
    parser = ArgumentParser()
    
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
    parser.add_argument('--max-features',action='store',type=int,default=5000,dest='max_features',
                        help='Maximum features to extract per image (default: 5000)')
    parser.add_argument('--ratio-threshold',action='store',type=float,default=0.75,dest='ratio_threshold',
                        help='Lowe ratio test threshold (default: 0.75)')
    parser.add_argument('--min-matches',action='store',type=int,default=50,dest='min_matches',
                        help='Minimum matches required between images (default: 50)') 
    parser.add_argument('--save-matches-vis', action='store_true', default=True, 
                        dest='save_matches_vis', help='whether to save images with matches drawn on them (default: True)')

    parser.add_argument('--fund-method',action='store',type=str,default='FM_RANSAC',
                        dest='fund_method',help='method to estimate fundamental matrix (default: FM_RANSAC)')
    parser.add_argument('--outlier-thres',action='store',type=float,default=3.0,
                        dest='outlier_thres',help='RANSAC outlier threshold for F-matrix estimation (default: 3.0)')
    parser.add_argument('--fund-prob',action='store',type=float,default=0.99,
                        dest='fund_prob',help='RANSAC confidence for F-matrix estimation (default: 0.99)')
    
    #misc
    parser.add_argument('--print-every',action='store', type=int, default=1, dest='print_every',
                        help='[1,+inf] print progress every print_every seconds, -1 to disable \
                        (default: 1)')
    parser.add_argument('--save-results',action='store_true',default=False, 
                        dest='save_results',help='whether to save images with\
                        keypoints drawn on them (default: False)')  
    
    return parser.parse_args()


if __name__=='__main__': 
    args = set_arguments()
    
    args.ext = [x for x in args.ext.split(',')]
    
    args.data_files_ = []
    if args.data_files != '': 
        args.data_files_ = args.data_files.split(',')
    args.data_files = args.data_files_
    
    if args.out_dir == '':
        args.out_dir = os.path.dirname(args.data_dir) if args.data_dir.endswith('/') else os.path.dirname(args.data_dir + '/')
        if args.out_dir == '':
            args.out_dir = '.'
    
    args.fund_method = getattr(cv2, args.fund_method)

    FeatMatch(args)
