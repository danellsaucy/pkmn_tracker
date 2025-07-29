import json
import cv2
import torch
import numpy as np
from PIL import Image
import imagehash
from pathlib import Path
from typing import List, Tuple, Optional
import mmcv
from mmdet.apis import init_detector, inference_detector


class PokemonCardScanner:
    def __init__(self, config_file: str, checkpoint_file: str, hash_dict_path: str):
        """
        Initialize the Pokemon Card Scanner
        
        Args:
            config_file: Path to model config file
            checkpoint_file: Path to model checkpoint
            hash_dict_path: Path to JSON file containing card hashes
        """
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = init_detector(config_file, checkpoint_file, device=self.device)
        
        # Load hash dictionary once
        with open(hash_dict_path, 'r') as f:
            self.hash_dict = json.load(f)
        
        # Card dimensions (Pokemon card ratio)
        self.out_h = 1024
        self.out_w = int(self.out_h * (6.3 / 8.8))  # ≈ 733
        
        # Standard destination points for perspective transform
        self.dst_pts = np.array([
            [0, 0],
            [self.out_w - 1, 0],
            [self.out_w - 1, self.out_h - 1],
            [0, self.out_h - 1]
        ], dtype="float32")
    
    def _fill_internal_holes(self, mask: np.ndarray) -> np.ndarray:
        """Fill internal holes in mask while preserving border details"""
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return binary_mask
        
        largest_contour = max(contours, key=cv2.contourArea)
        filled_mask = np.zeros_like(binary_mask)
        cv2.fillPoly(filled_mask, [largest_contour], 255)
        
        external_areas = (binary_mask == 0) & (filled_mask == 255)
        result_mask = filled_mask.copy()
        
        h, w = result_mask.shape
        border_connected = np.zeros_like(binary_mask)
        
        # Flood fill from border pixels
        for i in range(h):
            for j in range(w):
                if (i == 0 or i == h-1 or j == 0 or j == w-1) and external_areas[i, j]:
                    if border_connected[i, j] == 0:
                        cv2.floodFill(border_connected, None, (j, i), 255, 
                                     maskImage=external_areas.astype(np.uint8) * 255)
        
        result_mask[border_connected == 255] = 0
        return result_mask
    
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order points as top-left, top-right, bottom-right, bottom-left"""
        pts = pts.astype(np.float32)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).flatten()

        ordered = np.zeros((4, 2), dtype="float32")
        ordered[0] = pts[np.argmin(s)]      # top-left
        ordered[2] = pts[np.argmax(s)]      # bottom-right
        ordered[1] = pts[np.argmin(diff)]   # top-right
        ordered[3] = pts[np.argmax(diff)]   # bottom-left
        return ordered
    
    def _find_card_corners(self, mask: np.ndarray) -> np.ndarray:
        """Find the 4 corners of the card from the mask"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No contours found in mask")
        
        largest_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.01 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        if len(approx) >= 4:
            if len(approx) == 4:
                corner_points = approx.reshape(4, 2)
            else:
                # Find 4 extreme corners from multiple points
                points = approx.reshape(-1, 2)
                top_left = points[np.argmin(points.sum(axis=1))]
                bottom_right = points[np.argmax(points.sum(axis=1))]
                top_right = points[np.argmin(points[:, 1] - points[:, 0])]
                bottom_left = points[np.argmax(points[:, 1] - points[:, 0])]
                corner_points = np.array([top_left, top_right, bottom_right, bottom_left])
            
            return self._order_points(corner_points)
        else:
            # Fallback to bounding box
            white_pixels = np.where(mask > 0)
            if len(white_pixels[0]) == 0:
                raise ValueError("No white pixels found in mask")
            
            y_coords, x_coords = white_pixels
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()
            
            return np.array([
                [x_min, y_min], [x_max, y_min],
                [x_max, y_max], [x_min, y_max]
            ], dtype="float32")
    
    def _match_card_hash(self, warped_image: np.ndarray, top_k: int = 10) -> List[Tuple[str, int]]:
        """Match the warped card image against the hash database"""
        # Convert to PIL for hashing
        warped_rgb = cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB)
        warped_pil = Image.fromarray(warped_rgb)
        query_hash = imagehash.phash(warped_pil, hash_size=8)
        
        # Calculate distances
        distances = []
        for key, stored_hash_str in self.hash_dict.items():
            stored_hash = imagehash.hex_to_hash(stored_hash_str)
            dist = query_hash - stored_hash
            distances.append((key, dist))
        
        # Sort and return top matches
        distances.sort(key=lambda x: x[1])
        return distances[:top_k]
    
    def scan_card(self, 
                  image_path: str, 
                  output_path: Optional[str] = None,
                  top_k: int = 10,
                  pad: int = 0,
                  show_debug: bool = False) -> Tuple[np.ndarray, List[Tuple[str, int]]]:
        """
        Scan and identify a Pokemon card from an image
        
        Args:
            image_path: Path to input image
            output_path: Path to save warped card (optional)
            top_k: Number of top matches to return
            pad: Padding around detected card area
            show_debug: Whether to show debug images
            
        Returns:
            Tuple of (warped_card_image, list_of_matches)
            matches format: [(card_name, distance), ...]
        """
        # Load and infer
        image = mmcv.imread(image_path, channel_order='rgb')
        result = inference_detector(self.model, image)
        
        # Extract predictions
        masks = result.pred_instances.masks
        labels = result.pred_instances.labels
        scores = result.pred_instances.scores
        
        # Find Pokemon cards (class 0)
        card_indices = (labels == 0).nonzero(as_tuple=False).squeeze()
        
        if card_indices.numel() == 0:
            raise ValueError("No Pokemon card detected in image")
        
        if card_indices.ndim == 0:
            card_indices = card_indices.unsqueeze(0)
        
        # Choose best scoring detection
        best_idx = card_indices[scores[card_indices].argmax()]
        mask = masks[best_idx].cpu().numpy().astype(np.uint8) * 255
        
        # Smooth the mask
        kernel = np.ones((5, 5), np.uint8)
        mask_smooth = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_smooth = cv2.GaussianBlur(mask_smooth, (5, 5), 0)
        
        # Find bounding box
        ys, xs = np.where(mask_smooth > 0)
        if len(xs) == 0 or len(ys) == 0:
            raise ValueError("Empty mask detected")
        
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        
        # Apply padding with boundary checks
        x1 = max(x_min - pad, 0)
        x2 = min(x_max + pad, image.shape[1] - 1)
        y1 = max(y_min - pad, 0)
        y2 = min(y_max + pad, image.shape[0] - 1)
        
        # Crop image and mask
        cropped_image = image[y1:y2, x1:x2]
        mask_cropped = mask_smooth[y1:y2, x1:x2]
        mask_cropped = self._fill_internal_holes(mask_cropped)
        
        if show_debug:
            cv2.imshow("Original Cropped", cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
            cv2.imshow("Cleaned Mask", mask_cropped)
            cv2.waitKey(1000)
        
        # Find corners and apply perspective transform
        src_pts = self._find_card_corners(mask_cropped)
        M = cv2.getPerspectiveTransform(src_pts, self.dst_pts)
        
        # Warp the original cropped image
        cropped_bgr = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
        warped_final = cv2.warpPerspective(cropped_bgr, M, (self.out_w, self.out_h),
                                          borderMode=cv2.BORDER_CONSTANT,
                                          borderValue=(255, 255, 255))
        
        # Match against hash database
        matches = self._match_card_hash(warped_final, top_k)
        
        # Save output if requested
        if output_path:
            cv2.imwrite(output_path, warped_final)
            print(f"✅ Saved warped card to: {output_path}")
        
        # if show_debug:
        cv2.imshow("Warped Card", warped_final)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
        return warped_final, matches


def scan_pokemon_card(image_path: str,
                     config_file: str,
                     checkpoint_file: str, 
                     hash_dict_path: str,
                     output_path: Optional[str] = None,
                     top_k: int = 10) -> Tuple[np.ndarray, List[Tuple[str, int]]]:
    """
    Convenience function to scan a single Pokemon card
    
    Args:
        image_path: Path to input image
        config_file: Path to model config
        checkpoint_file: Path to model checkpoint  
        hash_dict_path: Path to hash dictionary JSON
        output_path: Path to save result (optional)
        top_k: Number of top matches to return
        
    Returns:
        Tuple of (warped_image, matches_list)
    """
    scanner = PokemonCardScanner(config_file, checkpoint_file, hash_dict_path)
    return scanner.scan_card(image_path, output_path, top_k)


# Example usage:
if __name__ == "__main__":
    # Paths
    config_file = r'C:\Users\daforbes\Desktop\projects\models\mask\pointmask_transforms\my_config.py'
    checkpoint_file = r'C:\Users\daforbes\Desktop\projects\models\mask\pointmask_transforms\best_coco_segm_mAP_epoch_99.pth'
    hash_dict_path = r"C:\Users\daforbes\Desktop\projects\tcg_scanner\raw\card_hashes.json"
    image_path = r"C:\Users\daforbes\Downloads\s-l1600 (1).jpg"
    
    # Option 1: Use the class for multiple scans
    scanner = PokemonCardScanner(config_file, checkpoint_file, hash_dict_path)
    
    warped_card, matches = scanner.scan_card(
        image_path=image_path,
        output_path="warped_card_output.png",
        top_k=10,
        show_debug=False
    )
    
    print("🔍 Top matches:")
    for i, (card_name, distance) in enumerate(matches, 1):
        print(f"{i:>2}. {card_name} (distance={distance})")
    
    # Option 2: Use convenience function for single scan
    # warped_card, matches = scan_pokemon_card(
    #     image_path, config_file, checkpoint_file, hash_dict_path,
    #     output_path="result.png"
    # )