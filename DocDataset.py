class DocDataset(Dataset):
    def __init__(self, origin, classes, img_size, grayscale=False):
        self.items = []
        self.gt = []

        if os.path.exists(origin):
            self.origin = origin
            self.img_size = img_size
            self.grayscale = grayscale
            self.classes = np.array(classes)
            
            for cls_id, cls_name in enumerate(self.classes):                
                cls_path = os.path.join(origin, cls_name)
                if os.path.exists(cls_path) and os.path.isdir(cls_path):
                    # get items
                    files = glob.glob(f"{cls_path}/*")
                    self.items.extend(files)
                    self.gt.extend([cls_id] * len(files))
            
            self.gt = torch.tensor(self.gt, dtype=torch.int64)
            self.items = np.array(self.items)
            self.loaded = True
        else:
            self.loaded = False

    @staticmethod
    def img_path_to_input(image_path, img_size, grayscale):
        """
        loads image from file and transforms it to tensor of required size
        """
        image = io.imread(image_path, as_gray=False)
        image = cv2.resize(image, dsize=(img_size,img_size), interpolation=cv2.INTER_NEAREST)

        if not grayscale:
            image = gray2rgb(image)
            image = image.transpose((2, 0, 1))

        image = torch.from_numpy(image)
        image = image / 255.
        return image

    def __len__(self):
        return len(self.items)
            
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # get item info
        image_path = self.items[idx]
        gt = self.gt[idx]
        # load and preprocess image
        image = DocDataset.img_path_to_input(image_path, self.img_size, self.grayscale)
        return image, gt, torch.tensor(idx, dtype=torch.long)