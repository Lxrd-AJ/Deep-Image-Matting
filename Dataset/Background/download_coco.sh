echo "Downloading the following images [COCO Dataset]"
# Verify no URL repeats more than once
sort COCO_Background.txt | uniq -c | sort -bgr

echo "Downloading images to './COCO_Images/'"
mkdir -p COCO_Images

wget -c -i COCO_Background.txt --directory-prefix=./COCO_Images/