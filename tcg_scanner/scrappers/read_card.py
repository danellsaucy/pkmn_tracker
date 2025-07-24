import cv2
import numpy as np
import utils


def readCard():
    pathImage = r"C:\Users\daforbes\Downloads\s-l1600 (1).jpg"
    #cam = cv2.VideoCapture(1)   # Use Video source 1 = phone; 0 = computer webcam

    # Scaled to the IRL height and width of a Pokemon card (6.6 cm x 8.8 cm)
    widthCard = 330
    heightCard = 440

    # Create a blank image
    blackImg = np.zeros((heightCard, widthCard, 3), np.uint8)
    # Read in picture and resize it to normalize
    pic = cv2.imread(pathImage)
    rot90frame = pic.copy()

    # Make image gray scale
    grayFrame = cv2.cvtColor(rot90frame, cv2.COLOR_BGR2GRAY)
    # Blur the image to reduce noise
    blurredFrame = cv2.GaussianBlur(grayFrame, (3, 3), 0)

    # Use Canny edge detection to get edges
    edgedFrame = cv2.Canny(image=blurredFrame, threshold1=100, threshold2=200)

    # Clean up edges
    kernel = np.ones((5,5))
    frameDial = cv2.dilate(edgedFrame, kernel, iterations=2)
    frameThreshold = cv2.erode(frameDial, kernel, iterations=1)

    # Get image contours
    contourFrame = rot90frame.copy()
    bigContour = rot90frame.copy()
    contours, hierarchy = cv2.findContours(frameThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contourFrame, contours, -1, (0, 255, 0), 10)
    cv2.namedWindow("Contours", cv2.WINDOW_NORMAL)  # Optional: allows resizing
    cv2.imshow("Contours", contourFrame)            # Show the image
    cv2.waitKey(0)                                   # Wait for a key press
    cv2.destroyAllWindows()      
    imgWarpColored = blackImg  # Set imgWarpColored
    # Get biggest contour
    corners, maxArea = utils.biggestContour(contours)
    print(f"corners: {corners}")
    print(f"maxArea: {maxArea}")
    if len(corners) == 4:
        corners = [corners[0][0], corners[1][0], corners[2][0], corners[3][0]]
        corners = utils.reorderCorners(corners)  # Reorders corners to [topLeft, topRight, bottomLeft, bottomRight]
        #cv2.drawContours(bigContour, corners, -1, (0, 255, 0), 10)
        bigContour = utils.drawRectangle(bigContour, corners)

        colors = [
            (0, 0, 255),    # Red     - Top-left
            (0, 255, 0),    # Green   - Top-right
            (255, 0, 0),    # Blue    - Bottom-right
            (0, 255, 255)   # Yellow  - Bottom-left
        ]

        corners = [tuple(map(int, pt[0])) for pt in corners]
        # Draw each point with its corresponding color
        for i, pt in enumerate(corners):
            cv2.circle(bigContour, tuple(map(int, pt)), 10, colors[i], -1)

        cv2.namedWindow("With Points", cv2.WINDOW_NORMAL)  # Optional: allows resizing
        cv2.imshow("With Points", bigContour)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        pts1 = np.float32([
            corners[0],  # Top-left
            corners[1],  # Top-right
            corners[2],  # Bottom-right
            corners[3],  # Bottom-left
        ])

        # Map to a perfect vertical card
        pts2 = np.float32([
            [0, 0],
            [widthCard, 0],
            [widthCard, heightCard],
            [0, heightCard]
        ])
        # Makes a matrix that transforms the detected card to a vertical rectangle
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        # Transforms card to a rectangle widthCard x heightCard
        imgWarpColored = cv2.warpPerspective(rot90frame, matrix, (widthCard, heightCard))

    # Resize all of the images to the same dimensions
    # Note: imgWarpColored is already resized and matchingCard gets resized in utils.getMatchingCard()
    rot90frame = cv2.resize(rot90frame, (widthCard, heightCard))
    grayFrame = cv2.resize(grayFrame, (widthCard, heightCard))
    blurredFrame = cv2.resize(blurredFrame, (widthCard, heightCard))
    edgedFrame = cv2.resize(edgedFrame, (widthCard, heightCard))
    contourFrame = cv2.resize(contourFrame, (widthCard, heightCard))
    bigContour = cv2.resize(bigContour, (widthCard, heightCard))

    cv2.namedWindow("imgWarpColored", cv2.WINDOW_NORMAL)
    cv2.imshow("imgWarpColored", imgWarpColored)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    import json
    from PIL import Image
    import imagehash

    # Load previously computed hashes
    with open(r"C:\Users\daforbes\Desktop\projects\card_hashes_phash_8x8.json", "r") as f:
        phash_dict = json.load(f)
    with open(r"C:\Users\daforbes\Desktop\projects\card_hashes_ahash_8x8.json", "r") as f:
        ahash_dict = json.load(f)
    with open(r"C:\Users\daforbes\Desktop\projects\card_hashes_dhash_8x8.json", "r") as f:
        dhash_dict = json.load(f)
    with open(r"C:\Users\daforbes\Desktop\projects\card_hashes_whash_8x8.json", "r") as f:
        whash_dict = json.load(f)

    # Convert OpenCV BGR image to PIL RGB
    warped_rgb = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2RGB)
    warped_pil = Image.fromarray(warped_rgb)

    # Generate perceptual hash
    pquery_hash = imagehash.phash(warped_pil, hash_size=8)
    #aquery_hash = imagehash.average_hash(warped_pil, hash_size=8)
    dquery_hash = imagehash.dhash(warped_pil, hash_size=8)
    #wquery_hash = imagehash.whash(warped_pil, hash_size=8)
    # Collect all distances
    pdistances = []
    adistances = []
    ddistances = []
    wdistances = []

    for key, stored_hash_str in phash_dict.items():
        stored_phash = imagehash.hex_to_hash(stored_hash_str)
        dist = pquery_hash - stored_phash  # Hamming distance
        pdistances.append((key, dist))
    
    # for key, stored_hash_str in ahash_dict.items():
    #     stored_ahash = imagehash.hex_to_hash(stored_hash_str)
    #     dist = aquery_hash - stored_ahash  # Hamming distance
    #     adistances.append((key, dist))

    for key, stored_hash_str in dhash_dict.items():
        stored_dhash = imagehash.hex_to_hash(stored_hash_str)
        dist = dquery_hash - stored_dhash  # Hamming distance
        ddistances.append((key, dist))

    # for key, stored_hash_str in whash_dict.items():
    #     stored_whash = imagehash.hex_to_hash(stored_hash_str)
    #     dist = wquery_hash - stored_whash  # Hamming distance
    #     wdistances.append((key, dist))

    # Sort by distance (lowest = best match)
    pdistances.sort(key=lambda x: x[1])
    # adistances.sort(key=lambda x: x[1])
    ddistances.sort(key=lambda x: x[1])
    # wdistances.sort(key=lambda x: x[1])

    # Convert to dicts for easier lookup
    phash_dict = dict(pdistances)
    dhash_dict = dict(ddistances)

    # Find keys that appear in both
    common_keys = set(phash_dict) & set(dhash_dict)

    # Combine distances (e.g., simple sum or weighted average)
    combined = [
        (key, phash_dict[key], dhash_dict[key], phash_dict[key] + dhash_dict[key])
        for key in common_keys
    ]

    # Sort by combined distance (lower is better)
    combined.sort(key=lambda x: x[3])

    # Print results
    print("🔗 Best matches (phash + dhash):")
    for i, (key, p_dist, d_dist, total) in enumerate(combined[:10]):
        print(f"{i+1:2}. {key} → phash: {p_dist}, dhash: {d_dist}, total: {total}")


    # # Show top 10 matches
    # print("🔍 Top 10 P HASH matches:")
    # for i, (key, dist) in enumerate(pdistances[:10]):
    #     print(f"{i+1:>2}. {key} (distance={dist})")
    
    # print("🔍 Top 10 A HASH matches:")
    # for i, (key, dist) in enumerate(adistances[:10]):
    #     print(f"{i+1:>2}. {key} (distance={dist})")

    # print("🔍 Top 10 D HASH matches:")
    # for i, (key, dist) in enumerate(ddistances[:10]):
    #     print(f"{i+1:>2}. {key} (distance={dist})")

    # print("🔍 Top 10 W HASH matches:")
    # for i, (key, dist) in enumerate(wdistances[:10]):
    #     print(f"{i+1:>2}. {key} (distance={dist})")
    # # Check if a matching card has been found, and if so, display it
    # found, matchingCard = utils.findCard(imgWarpColored.copy())  # Check to see if a matching card was found

    # # An array of all 8 images
    # imageArr = ([rot90frame, grayFrame, blurredFrame, edgedFrame],
    #             [contourFrame, bigContour, imgWarpColored, matchingCard])

    # # Labels for each image
    # labels = [["Original", "Gray", "Blurred", "Threshold"],
    #           ["Contours", "Biggest Contour", "Warped Perspective", "Matching Card"]]

    # # Stack all 8 images into one and add text labels
    # stackedImage = utils.makeDisplayImage(imageArr, labels)

    # # Display the image
    # cv2.imshow("Card Finder", stackedImage)


if __name__ == '__main__':
    readCard()  # Finds and reads from a saved image or live feed
