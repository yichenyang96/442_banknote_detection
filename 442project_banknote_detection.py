import numpy as np
import cv2
import sys
import getopt
import wave
import pyaudio
import os
import gzip
import struct
from sklearn.preprocessing import normalize
import collections


def get_value(name):
    value_pos = name.find("_")
    value = name[0: value_pos]
    rest1 = name[value_pos + 1:]
    back_front_pos = rest1.find("_")
    back_front = rest1[:back_front_pos]
    rest2 = rest1[back_front_pos + 1:]
    begin_pos = rest2.find("_")
    end_pos = rest2.find(".")
    country = rest2[begin_pos + 1:end_pos]
    if back_front == "0":
        side = "front"
    else:
        side = "back"
    out = country + " " + side + " " + value
    return out


def remove_kp(kp, des, pts):
    remain_kp = []
    remain_des = []
    # print("Removing...")
    for i in range(len(kp)):
        if np.float32([kp[i].pt])[0].tolist() not in pts:
            remain_kp.append(kp[i])
            remain_des.append(des[i])
    return remain_kp, np.array(remain_des)


def sift(path):
    name = path

    MIN_POINT = 25

    cv2.useOptimized()
    detector = cv2.xfeatures2d.SIFT_create()
    norm = cv2.NORM_L1
    matcher = cv2.BFMatcher(norm)
    # matcher = cv2.BFMatcher()

    # des_list = np.load("video_des_list_gray.npy")
    # name_list = np.load("video_name_list_gray.npy")
    # kp_list = np.load("video_kp_list_gray.npy")
    # image_list = np.load("video_image_list_gray.npy")

    des_list = []
    name_list = []
    kp_list = []
    img_list = []

    for eachphoto in os.listdir("all_money_new_back"):
        image = cv2.imread("all_money_new_back/" + eachphoto)
        image = cv2.resize(image, (300, 140))
        img_list.append(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp_curr, des_curr = detector.detectAndCompute(image, None)
        des_list.append(des_curr)
        name_list.append(get_value(eachphoto))
        kp_list.append(kp_curr)

    # for i in range(16, 26):
    #     name = "test" + str(i) + ".jpg"
    #     out = "hiddenout" + str(i) + ".jpg"
    img2 = cv2.imread(name)
    h_img2, w_img2 = img2.shape[:2]
    h_curr = int(float(1000) / float(w_img2) * float(h_img2))
    img2_curr = cv2.resize(img2, (1000, h_curr))
    img2 = cv2.cvtColor(img2_curr, cv2.COLOR_BGR2GRAY)
    keep_continue = True

    # use the label part
    # name_list_label = {}
    # des_list_label = {}
    # kp_list_label = {}
    # img_list_label = {}
    # for eachnum in os.listdir("number"):
    #     curr_des_list = []
    #     curr_name_list = []
    #     curr_kp_list = []
    #     curr_img_list = []
    #     path_curr = "number/" + eachnum
    #     for eachphoto in os.listdir(path_curr):
    #         image = cv2.imread(path_curr + "/" + eachphoto)
    #         image = cv2.resize(image, (300, 140))
    #         curr_img_list.append(image)
    #         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #         kp_curr, des_curr = detector.detectAndCompute(image, None)
    #         curr_des_list.append(des_curr)
    #         curr_name_list.append(get_value(eachphoto))
    #         curr_kp_list.append(kp_curr)
    #     name_list_label[eachnum] = curr_name_list
    #     des_list_label[eachnum] = curr_des_list
    #     kp_list_label[eachnum] = curr_kp_list
    #     img_list_label[eachnum] = curr_img_list

    # Capture frame-by-frame
    h1, w1 = 140, 300
    h3, w2 = img2.shape[:2]
    h2 = h3 + 200
    vis = np.zeros((max(h1, h2), w1 + w2), np.uint8)
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    find_one = False
    count = 0
    print("finished")
    # kp2_curr, desc2_curr = detector.detectAndCompute(img2, None)
    while keep_continue:
        print(count)
        if count > 5:
            break
        keep_continue = False
        img1 = img2
        showText = ""
        p1 = []
        p2 = []

        # tf, label = digit_total(img2)
        # if tf:
        #     name_list_curr = name_list_label[label]
        #     img_list_curr = img_list_label[label]
        #     kp_list_curr = kp_list_label[label]
        #     des_list_curr = des_list_label[label]
        # else:
        #     name_list_curr = name_list
        #     img_list_curr = img_list
        #     kp_list_curr = kp_list
        #     des_list_curr = des_list

        # name_list_curr = name_list
        # img_list_curr = img_list
        # kp_list_curr = kp_list
        # des_list_curr = des_list

        kp2_curr, desc2_curr = detector.detectAndCompute(img2, None)
        # for searchIndex in range(len(name_list_curr)):
        #     img1_curr = img_list_curr[searchIndex]
        #     kp1_curr = kp_list_curr[searchIndex]
        #     desc1_curr = des_list_curr[searchIndex]
        #     showText_curr = name_list_curr[searchIndex]

        for searchIndex in range(len(name_list)):
            img1_curr = img_list[searchIndex]
            kp1_curr = kp_list[searchIndex]
            desc1_curr = des_list[searchIndex]
            showText_curr = name_list[searchIndex]

            # matching feature
            raw_matches = matcher.knnMatch(desc1_curr, trainDescriptors=desc2_curr, k=2)  # 2
            p1_curr, p2_curr, kp_pairs_curr = filter_matches(kp1_curr, kp2_curr, raw_matches)
            if len(p1_curr) > len(p1):
                img1 = img1_curr
                showText = showText_curr
                p1 = p1_curr
                p2 = p2_curr

        if len(p1) >= MIN_POINT:
            keep_continue = True
            if not find_one:
                find_one = True
                vis[:h3, w1:w1 + w2] = img2_curr

            if count * h1 + h1 <= h2:
                vis[count * h1:h1 + count * h1, :w1] = img1

            H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
            if H is not None:
                corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
                corners = np.int32(
                    cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0))
                cv2.polylines(vis, [corners], True, (0, 255, 0))
                cv2.fillPoly(img2, [corners - [300, 0]], (0, 0, 0))
                # kp2_curr, desc2_curr = remove_kp(kp2_curr, desc2_curr, p2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                point1 = int((corners[0][0] + corners[2][0]) / 2)
                point2 = int((corners[0][1] + corners[2][1]) / 2)
                cv2.putText(vis, showText, (point1, point2), font, 1, (0, 0, 255), 3)

            for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
                if inlier:
                    col = (0, 255, 0)
                    if count * h1 + h1 <= h2:
                        cv2.circle(vis, (int(x1), int(y1 + count * h1)), 2, col, -1)
                    cv2.circle(vis, (int(x2 + w1), int(y2)), 2, col, -1)
        count += 1
    cv2.imwrite("output.jpg", vis)
    cv2.imshow('442_project_detect_banknote', vis)
    while True:
        k = cv2.waitKey(1000)
        if k == 27:
            break
    cv2.destroyAllWindows()
        # cv2.imwrite(out, vis)



def filter_matches(kp1, kp2, matches, ratio=0.75):
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append(kp1[m.queryIdx])
            mkp2.append(kp2[m.trainIdx])
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    # print(mkp2)
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, kp_pairs


def video_detect():
    MIN_POINT = 30

    title_img = cv2.imread("title.png")
    # title_img = cv2.cvtColor(title_img, cv2.COLOR_BGR2GRAY)
    h_title, w_title = title_img.shape[:2]

    cv2.useOptimized()
    cap = cv2.VideoCapture(0)
    detector = cv2.xfeatures2d.SIFT_create()
    norm = cv2.NORM_L1
    matcher = cv2.BFMatcher(norm)
    # matcher = cv2.BFMatcher()

    # des_list = np.load("video_des_list_gray.npy")
    # name_list = np.load("video_name_list_gray.npy")
    # kp_list = np.load("video_kp_list_gray.npy")
    # image_list = np.load("video_image_list_gray.npy")

    des_list = []
    name_list = []
    kp_list = []
    img_list = []

    for eachphoto in os.listdir("all_money_new"):
        image_origin = cv2.imread("all_money_new/" + eachphoto)
        image_origin = cv2.resize(image_origin, (300, 140))
        image = cv2.cvtColor(image_origin, cv2.COLOR_BGR2GRAY)
        img_list.append(image_origin)
        kp_curr, des_curr = detector.detectAndCompute(image, None)
        des_list.append(des_curr)
        name_list.append(get_value(eachphoto))
        kp_list.append(kp_curr)

    # np.save("video_des_list_gray", des_list)
    # np.save("video_name_list_gray", name_list)
    # np.save("video_kp_list_gray", kp_list)
    # np.save("video_image_list_gray.npy", img_list)
    h1, w1 = 140, 300
    h2, w2 = 700, 640
    h3 = 480
    vis_final = np.zeros((max(h1, h2) + h_title, w1 + w2), np.uint8)
    vis_final = cv2.cvtColor(vis_final, cv2.COLOR_GRAY2BGR)
    vis_final[0:h_title, :] = title_img
    print("finished")
    searchIndex = 0
    name_list_length = len(name_list)
    detected_list = []

    while True:
        length_detected_list = len(detected_list)
        # make sure the searchIndex is not too large
        if searchIndex >= name_list_length:
            searchIndex = searchIndex % name_list_length

        # read in the audio
        p = pyaudio.PyAudio()
        # get the image
        ret, frame_img = cap.read()
        img2 = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)
        # kp2, desc2 = detector.detectAndCompute(img2, None)

        # Capture frame-by-frame
        vis = np.zeros((max(h1, h2), w1 + w2), np.uint8)
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        vis[0:h3, w1:w1 + w2] = frame_img

        if length_detected_list > 0:
            for i in range(length_detected_list):
                img1 = img_list[detected_list[i]]
                kp1 = kp_list[detected_list[i]]
                desc1 = des_list[detected_list[i]]
                showText = name_list[detected_list[i]]
                if i * h1 + h1 <= h2:
                    vis[i * h1:h1 + i * h1, :w1] = img1

                # calculate features
                kp2, desc2 = detector.detectAndCompute(img2, None)
                # matching feature
                raw_matches = matcher.knnMatch(desc1, trainDescriptors=desc2, k=2)  # 2
                p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)

                # find whether it is right one
                if len(p1) >= MIN_POINT:
                    H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
                    if H is not None:
                        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
                        corners = np.int32(
                            cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0))
                        cv2.polylines(vis, [corners], True, (0, 255, 0))
                        cv2.fillPoly(img2, [corners - [300, 0]], (0, 0, 0))
                        # kp2, desc2 = remove_kp(kp2, desc2, p2)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        point1 = int(float(corners[0][0] + corners[2][0]) / 2.0)
                        point2 = int(float(corners[0][1] + corners[2][1]) / 2.0)
                        cv2.putText(vis, showText, (point1, point2), font, 1, (0, 0, 255), 2)

                    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
                        if inlier:
                            col = (0, 255, 0)
                            if i * h1 + h1 <= h2:
                                cv2.circle(vis, (int(x1), int(y1 + i * h1)), 2, col, -1)
                            cv2.circle(vis, (int(x2 + w1), int(y2)), 2, col, -1)
                else:
                    detected_list = detected_list[0:i]
                    vis_final[h_title:, :] = vis
                    cv2.imshow('442_project_detect_banknote', vis_final)
                    break

        img1 = img_list[searchIndex]
        kp1 = kp_list[searchIndex]
        desc1 = des_list[searchIndex]
        showText = name_list[searchIndex]
        if length_detected_list * h1 + h1 <= h2:
            vis[length_detected_list * h1:h1 + length_detected_list * h1, :w1] = img1

        # calculate features
        kp2, desc2 = detector.detectAndCompute(img2, None)
        # matching feature
        raw_matches = matcher.knnMatch(desc1, trainDescriptors=desc2, k=2)  # 2
        p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)

        # find whether it is right one
        if len(p1) >= MIN_POINT:
            # if "china" in showText:
            #     if "front" in showText:
            #         p1, p2 = mask("china", img2)

            detected_list.append(searchIndex)
            H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
            if H is not None:
                corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
                corners = np.int32(
                    cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0))
                cv2.polylines(vis, [corners], True, (0, 255, 0))
                cv2.fillPoly(img2, [corners - [300, 0]], (0, 0, 0))
                font = cv2.FONT_HERSHEY_SIMPLEX
                point1 = int(float(corners[0][0] + corners[2][0])/2.0)
                point2 = int(float(corners[0][1] + corners[2][1]) / 2.0)
                cv2.putText(vis, showText, (point1, point2), font, 1, (0, 0, 255), 2)

            for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
                if inlier:
                    col = (0, 255, 0)
                    if length_detected_list * h1 + h1 <= h2:
                        cv2.circle(vis, (int(x1), int(y1 + length_detected_list * h1)), 2, col, -1)
                    cv2.circle(vis, (int(x2 + w1), int(y2)), 2, col, -1)

        # here is used to make sure the vis_final has right size
        # if len(vis.shape) < len(vis_final.shape):
        #     vis_final = cv2.cvtColor(vis_final, cv2.COLOR_RGB2GRAY)
        # elif len(vis.shape) > len(vis_final.shape):
        #     vis_final = cv2.cvtColor(vis_final, cv2.COLOR_GRAY2BGR)

        # print vis_final
        vis_final[h_title:, :] = vis
        cv2.imshow('442_project_detect_banknote', vis_final)

        searchIndex += 1

        if cv2.waitKey(1) == 27:
            break

    # close PyAudio
    p.terminate()

    cap.release()
    cv2.destroyAllWindows()


def read_idx(mnist_filename):
    with gzip.open(mnist_filename) as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I',
                                    f.read(4))[0] for d in range(dims))
        data_as_array = np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
        return data_as_array


def run(gray_in):
    mnist_train_images2 = read_idx("jpg_to_mnist/single-images-idx3-ubyte.gz")
    mnist_train_labels2 = read_idx("jpg_to_mnist/single-labels-idx1-ubyte.gz")

    number2 = mnist_train_images2.shape[0]
    height2 = mnist_train_images2.shape[1]
    width2 = mnist_train_images2.shape[2]
    X2 = np.ones((height2 * width2, number2))

    for i in range(number2):
        norm_image2 = np.ones(height2 * width2)
        norm_image2 = normalize(mnist_train_images2[i]) * 255
        image_pixel2 = (norm_image2).reshape(height2 * width2)
        X2[:, i] = image_pixel2

    mean2 = X2.mean(1).reshape((height2 * width2, 1))
    mean_matrix2 = np.tile(mean2, (1, number2))
    X2 = X2 - mean_matrix2
    print(X2.shape)
    # XXT2 = np.matmul(X2, np.transpose(X2))
    # print(XXT2.shape)
    #
    # w2, v2 = eigh(XXT2, eigvals=(764, 783))
    #
    # print(v2.shape)

    covariance = np.dot(X2, X2.T)
    w2, v2 = np.linalg.eig(covariance)
    w2 = w2[0:20]
    v2 = (v2.T[0:20]).T

    W_train2 = np.ones((number2, 20))
    for i in range(number2):
        image_pixel_train2 = normalize(mnist_train_images2[i].reshape((1, height2 * width2))) - mean2.reshape(1,
                                                                                                              height2 * width2)
        W_train2[i] = np.matmul(image_pixel_train2, v2)

    im = gray_in
    im = cv2.resize(im, (465, 195))

    ret, im_th = cv2.threshold(im, 90, 255, cv2.THRESH_BINARY_INV)
    _, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]

    single_digit = {}

    for rect in rects:
        # Detect a single digit: 1,2,5,0 and then connect
        if rect[2] < 75 and rect[3] < 80 and rect[2] > 5 and rect[3] > 20:  # rec[3] height
            roi = im_th[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
            roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            single_digit[rect] = {}
            single_digit[rect]["roi"] = roi
    single_digit = collections.OrderedDict(sorted(single_digit.items()))
    key_list = []
    for key in single_digit:
        key_list.append(key)

    digits = []
    skip_value = -1
    if key_list:
        # find left digit
        for i in range(len(key_list)):
            if i < skip_value:
                continue
            else:
                rect = key_list[i]
                right_x = rect[0] + rect[2]
                right_top_y = rect[1]
                right_bottom_y = rect[1] + rect[3]
            for j in range(len(key_list[i + 1:])):
                rect2 = key_list[j + i + 1]
                left_x = rect2[0]
                left_top_y = rect2[1]
                left_bottom_y = rect2[1] + rect2[3]
                if abs(right_x - left_x) < 6 and abs(right_top_y - left_top_y) < 6 and abs(
                        right_bottom_y - left_bottom_y) < 6:
                    print("find nearby points")
                    skip_value = j + i + 1
                    if i not in digits:
                        digits.append(i)
                    if j + i + 1 not in digits:
                        digits.append(j + i + 1)
                    break
    print(digits)
    # check if it is a correct digit
    islegal = True
    label = ""
    if not digits:
        return False, ""
    else:
        rect0 = key_list[digits[0]]
        image_pixel_test = normalize(single_digit[rect0]["roi"]).reshape(1, height2 * width2) - mean2.reshape(1,
                                                                                                              height2 * width2)
        W_test = np.matmul(image_pixel_test, v2)
        single_label = str(KNN(W_test[0], W_train2, mnist_train_labels2, mnist_train_images2))
        if len(digits) == 1:
            islegal = False
        if str(single_label) == "0":
            print("first digit is 0")
            islegal = False
        else:
            label += single_label
        for digit in digits[1:]:
            rect_remain = key_list[digit]
            image_pixel_test = normalize(single_digit[rect_remain]["roi"]).reshape(1, height2 * width2) - mean2.reshape(
                1, height2 * width2)
            W_test = np.matmul(image_pixel_test, v2)
            single_label = str(KNN(W_test[0], W_train2, mnist_train_labels2, mnist_train_images2))
            print(single_label)
            if str(single_label) != "0":
                print("remaining digit is not 0")
                islegal = False
            else:
                label += single_label
        if islegal == True:
            # cv2.rectangle(im, (rect0[0], rect0[1]), (rect_remain[0] + rect_remain[2], rect_remain[1] + rect_remain[3]),
            #               (0, 255, 0), 3)
            # cv2.putText(im, str(label), (rect0[0], rect0[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
            #
            return True, label
        else:
            return False, ""
    # cv2.imshow("Resulting Image with Rectangular ROIs", im)
    # cv2.waitKey()


def KNN(W_test, W_train, mnist_train_labels, mnist_train_images):
    distanceArray = np.ones(W_train.shape[0])
    for i in range(W_train.shape[0]):
        dis = calculateSquareDistance(W_test, W_train[i])
        distanceArray[i] = dis
    minPos = np.argsort(distanceArray)[:1]
    labels = mnist_train_labels[minPos]
    cv2.imshow("xxxx_{0}: ".format(minPos), mnist_train_images[minPos].reshape(28, 28))
    binCount = np.bincount(labels)
    print("bincount: ", binCount)
    label = np.argmax(binCount)
    return label


# It will return the square of distance
def calculateSquareDistance(p1, p2):
    diff = p1 - p2
    return np.sum(diff ** 2)


def pre_process(gray):
    # image = cv2.imread(path)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (2, 2))
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return gray


def digit_total(pic):
    gray = pre_process(pic)
    tf, label = run(gray)
    return tf, label


def main():
    input_type = input('Choose a function, 1 for image detect, 2 for video detect: ')
    type_int = int(input_type)
    if type_int == 1:
        image_path = input('please input the path of the image: ')
        sift(image_path)
        # sift("")
    elif type_int == 2:
        video_detect()
    else:
        print("Wrong arguments!")
        return -1


if __name__ == '__main__':
    main()
