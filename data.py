"""
PyTorch Video Dataset Class for loading videos using PyTorch
Dataloader. This Dataset assumes that video files are Preprocessed
 by being trimmed over time and resizing the frames.

Mohsen Fayyaz __ Sensifai Vision Group
http://www.Sensifai.com

If you find this code useful, please star the repository.
"""

from __future__ import print_function, division
import cv2
import sys
import os
import torch
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import torchvision.transforms.functional as F
from facenet_pytorch import MTCNN
from skimage import io
import dlib


class RandomCrop(object):
    """Crop randomly the frames in a clip.

	Args:
		output_size (tuple or int): Desired output size. If int, square crop
			is made.
	"""

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, clip):

        h, w = clip.size()[2:]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        clip = clip[:, :, top: top + new_h,
               left: left + new_w]

        return clip


class VideoDataset(Dataset):
    """Dataset Class for Loading Video"""

    def __init__(self, files, labels, num_frames, transform=None, test=False):
        """
		Args:
			clipsList (string): Path to the clipsList file with labels.
			rootDir (string): Directory with all the videoes.
			transform (callable, optional): Optional transform to be applied
				on a sample.
			channels: Number of channels of frames
			timeDepth: Number of frames to be loaded in a sample
			xSize, ySize: Dimensions of the frames
			mean: Mean valuse of the training set videos over each channel
		"""
        self.files = files
        self.labels  = labels
        self.num_frames = num_frames
        self.max_num_frames = 120
        self.transform = transform
        self.test = test
        self.frame_skip = 10
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def face_detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Resize frame of video to 1/4 size for faster face detection processing
        small_frame = cv2.resize(gray, (0, 0), fx=0.25, fy=0.25)
        # Detect the faces
        faces = self.face_cascade.detectMultiScale(small_frame, 1.1, 4)
        return faces


    def __len__(self):
        return len(self.files)

    def readVideo(self, videoFile):

        X = []
        X_nofaces = []
        # Load the cascade

        # Open the video file
        cap = cv2.VideoCapture(videoFile)
        cap.set(1, self.frame_skip)
        # nFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # frames = torch.FloatTensor(self.channels, self.timeDepth, self.xSize, self.ySize)
        face_detected = False
        max_attempts = 90
        attempts = 0
        while attempts < max_attempts:
            ret, frame = cap.read()
            attempts += 1
            if ret:
                try:
                    # detect faces
                    if not face_detected:

                        faces = self.face_detect(frame)
                        # Face detected
                        if len(faces) == 1:
                            # Get the first face
                            x, y, w, h = faces[0] * 4
                            face_detected = True
                        else:
                            frame = torch.from_numpy(frame)
                            # HWC2CHW
                            frame = frame.permute(2, 0, 1)
                            if self.transform is not None:
                                frame = F.to_pil_image(frame)
                                frame = self.transform(frame)
                            X_nofaces.append(frame.squeeze_(0))
                            if len(X_nofaces) > self.num_frames:
                                break

                    if face_detected:
                        face_img = frame[y: y + h, x: x+w]
                        frame = torch.from_numpy(face_img)
                        # HWC2CHW
                        frame = frame.permute(2, 0, 1)
                        if self.transform is not None:
                            frame = F.to_pil_image(frame)
                            frame = self.transform(frame)
                        X.append(frame.squeeze_(0))
                        if len(X) > self.num_frames:
                            break
                except:
                    pass
            else:
                break

        if len(X) > self.num_frames:
            X = torch.stack(X, dim=0)
            return X
        else:
            X_nofaces = torch.stack(X_nofaces, dim=0)
            return X_nofaces

    def __getitem__(self, index):

        file = self.files[index]
        X = self.readVideo(file)
        if self.test:
            y = self.labels[index]
        else:
            y = torch.LongTensor([self.labels[index]])  # (labels) LongTensor are for int64 instead of FloatTensor

        return X, y


class FrameDataset(Dataset):
    """Dataset Class for Loading Video"""

    def __init__(self, files, labels, num_frames, transform=None, test=False):
        """
        """
        self.files = files
        self.labels  = labels
        self.num_frames = num_frames
        self.max_num_frames = 60
        self.transform = transform
        self.test = test
        self.frame_no = num_frames
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def face_detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Resize frame of video to 1/4 size for faster face detection processing
        # small_frame = cv2.resize(gray, (0, 0), fx=0.25, fy=0.25)
        # Detect the faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return faces


    def __len__(self):
        return len(self.files)


    def readVideo(self, videoFile):

        # Load the cascade

        # Open the video file
        cap = cv2.VideoCapture(videoFile)
        # cap.set(1, self.frame_no)
        # nFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # frames = torch.FloatTensor(self.channels, self.timeDepth, self.xSize, self.ySize)

        attempts = 0
        while attempts < self.max_num_frames:
            ret, frame = cap.read()
            attempts += 1
            if ret:
                last_good_frame = frame
                try:
                    faces = self.face_detect(frame)
                    # Face detected
                    if len(faces) > 0:
                        # Get the face, if more than two, use the whole frame
                        if len(faces) > 1:
                            break
                        x, y, w, h = faces[0]
                        # x -= 40
                        # y -= 40
                        # w += 80
                        # h += 80
                        face_img = frame[y: y + h, x: x + w]
                        frame = torch.from_numpy(face_img)
                        # HWC2CHW
                        frame = frame.permute(2, 0, 1)
                        if self.transform is not None:
                            frame = F.to_pil_image(frame)
                            frame = self.transform(frame)
                            cap.release()
                            return frame
                except:
                    # print("Face detection error!")
                    pass
            else:
                break
        frame = torch.from_numpy(last_good_frame)
        # HWC2CHW
        frame = frame.permute(2, 0, 1)
        if self.transform is not None:
            frame = F.to_pil_image(frame)
            frame = self.transform(frame)
        cap.release()
        return frame


    def __getitem__(self, index):

        file = self.files[index]
        X = self.readVideo(file)
        if self.test:
            y = self.labels[index]
        else:
            y = torch.LongTensor([self.labels[index]])  # (labels) LongTensor are for int64 instead of FloatTensor

        return X, y


class FrameDataset_mtcnn(Dataset):
    """Dataset Class for Loading Video"""

    def __init__(self, files, labels, num_frames, transform=None, test=False):
        """
        """
        self.files = files
        self.labels  = labels
        self.num_frames = num_frames
        self.max_num_frames = 60
        self.transform = transform
        self.test = test
        self.frame_no = num_frames
        self.mtcnn =  MTCNN(margin=40, keep_all=True, post_process=False, device='cuda:0')


    # def face_detect(self, frame):
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     # Resize frame of video to 1/4 size for faster face detection processing
    #     # small_frame = cv2.resize(gray, (0, 0), fx=0.25, fy=0.25)
    #     # Detect the faces
    #     return faces


    def __len__(self):
        return len(self.files)


    def readVideo(self, videoFile):

        # Load the cascade

        # Open the video file
        cap = cv2.VideoCapture(videoFile)
        # cap.set(1, self.frame_no)
        # nFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # frames = torch.FloatTensor(self.channels, self.timeDepth, self.xSize, self.ySize)

        attempts = 0
        while attempts < self.max_num_frames:
            ret, frame = cap.read()
            attempts += 1
            if ret:
                last_good_frame = frame
                try:
                    faces = self.mtcnn(frame)
                    # Face detected
                    if len(faces) > 0:
                        # Get the face, if more than two, use the whole frame
                        if len(faces) > 1:
                            break
                        frame = faces[0]

                        if self.transform is not None:
                            frame = F.to_pil_image(frame)
                            frame = self.transform(frame)
                            cap.release()
                            return frame
                except:
                    # print("Face detection error!")
                    pass
            else:
                break
        frame = torch.from_numpy(last_good_frame)
        # HWC2CHW
        frame = frame.permute(2, 0, 1)
        if self.transform is not None:
            frame = F.to_pil_image(frame)
            frame = self.transform(frame)
        cap.release()
        return frame


    def __getitem__(self, index):

        file = self.files[index]
        X = self.readVideo(file)
        if self.test:
            y = self.labels[index]
        else:
            y = torch.LongTensor([self.labels[index]])  # (labels) LongTensor are for int64 instead of FloatTensor

        return X, y


class ImageDataset(Dataset):
    """Dataset Class for Loading Video"""

    def __init__(self, files, labels, num_frames, transform=None, test=False):
        """
        """
        self.files = files
        self.labels  = labels
        self.transform = transform
        self.test = test


    def __len__(self):
        return len(self.files)


    def get_b(self, img):

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(img_gray)

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        faces = detector(img_gray)
        if len(faces) > 0:
            face = faces[0]
            landmarks = predictor(img_gray, face)
            landmarks_points = []
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                landmarks_points.append((x, y))

                # cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

            points = np.array(landmarks_points, np.int32)
            convexhull = cv2.convexHull(points)
            # cv2.polylines(img, [convexhull], True, (255, 0, 0), 3)
            cv2.fillConvexPoly(mask, convexhull, 255)

            kernel = np.ones((5, 5), np.uint8)
            outer_b = cv2.dilate(mask, kernel, iterations=10)
            inner_b = cv2.erode(mask, kernel, iterations=10)

            mask_inv = cv2.bitwise_not(inner_b)
            mask = cv2.bitwise_and(outer_b, outer_b, mask=mask_inv)

            face_image_1 = cv2.bitwise_and(img, img, mask=mask)

            img = cv2.cvtColor(face_image_1, cv2.COLOR_BGR2RGB)
            return img
        else:
            return img

    def __getitem__(self, index):

        file = self.files[index]
        image = io.imread(file)
        image = self.get_b(image)

        if self.transform:
            image = F.to_pil_image(image)
            image = self.transform(image)
        if self.test:
            y = self.labels[index]
        else:
            y = torch.LongTensor([self.labels[index]])  # (labels) LongTensor are for int64 instead of FloatTensor

        return image, y