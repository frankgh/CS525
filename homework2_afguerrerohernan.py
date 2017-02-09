# NOTE -- please do NOT put your name(s) in the Python code; instead, name the Python file
# itself to include your WPI username(s).

# import matplotlib.pyplot as plot
import cv2  # Uncomment if you have OpenCV and want to run the real-time demo
import numpy as np


def J(w, faces, labels, alpha=0.):
    e = faces.dot(w) - labels
    r = alpha * w.T.dot(w) if alpha > 0. else 0
    return 0.5 * (e.T.dot(e) + r)


def gradJ(w, faces, labels, alpha=0.):
    return faces.T.dot(faces.dot(w) - labels) + (alpha * w)


def gradientDescent(trainingFaces, trainingLabels, testingFaces, testingLabels, alpha=0.):
    w1 = np.random.rand(trainingFaces.shape[1])

    epsilon_1 = 1e-6 * 8
    epsilon_2 = 1e-6 * 4
    epsilon_3 = 1e-6
    epsilon = epsilon_3
    delta = 1e-4
    n = 0

    while True:
        w2 = w1 - (epsilon * gradJ(w1, trainingFaces, trainingLabels, alpha))

        cost_w1 = J(w1, trainingFaces, trainingLabels, alpha)
        cost_w2 = J(w2, trainingFaces, trainingLabels, alpha)
        cost_change = np.abs(cost_w1 - cost_w2)

        norm = (w2 - w1)
        norm = np.sqrt(norm.T.dot(norm))

        # terrible impl of adaptive learning rate
        if cost_change > 4 * delta:
            epsilon = epsilon_1
        elif cost_change > 2.5 * delta:
            epsilon = epsilon_2
        else:
            epsilon = epsilon_3

        if n % 100 == 0:
            print 'norm: {:f}, cost: {:f}, iterations: {}'.format(norm, J(w2, trainingFaces, trainingLabels, alpha), n)

        w1 = w2
        n += 1

        if cost_change < delta:
            break
    return w2


def method1(trainingFaces, trainingLabels, testingFaces, testingLabels):
    A = trainingFaces.T.dot(trainingFaces)  # + (1e3 * np.eye(trainingFaces.shape[1]))
    y = trainingFaces.T.dot(trainingLabels)
    return np.linalg.solve(A, y)


def method2(trainingFaces, trainingLabels, testingFaces, testingLabels):
    return gradientDescent(trainingFaces, trainingLabels, testingFaces, testingLabels)


def method3(trainingFaces, trainingLabels, testingFaces, testingLabels):
    alpha = 837
    return gradientDescent(trainingFaces, trainingLabels, testingFaces, testingLabels, alpha)


def reportCosts(we, trainingFaces, trainingLabels, testingFaces, testingLabels, alpha=0.):
    print '{:>14} {:f}'.format('Training cost:', J(we, trainingFaces, trainingLabels, alpha))
    print '{:>14} {:f}'.format('Testing cost:', J(we, testingFaces, testingLabels, alpha))


# Accesses the web camera, displays a window showing the face, and classifies smiles in real time
# Requires OpenCV.
def detectSmiles(w):
    # Given the image captured from the web camera, classify the smile
    def classifySmile(im, imGray, faceBox, w):
        # Extract face patch as vector
        face = imGray[faceBox[1]:faceBox[1] + faceBox[3], faceBox[0]:faceBox[0] + faceBox[2]]
        face = cv2.resize(face, (24, 24))
        face = (face - np.mean(face)) / np.std(face)  # Normalize
        face = np.reshape(face, face.shape[0] * face.shape[1])

        # Classify face patch
        yhat = w.dot(face)
        # print yhat

        # Draw result as colored rectangle
        THICKNESS = 3
        green = 128 + (yhat - 0.5) * 255
        color = (0, green, 255 - green)
        pt1 = (faceBox[0], faceBox[1])
        pt2 = (faceBox[0] + faceBox[2], faceBox[1] + faceBox[3])
        cv2.putText(im, str(yhat), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.rectangle(im, pt1, pt2, color, THICKNESS)

    # Starting video capture
    vc = cv2.VideoCapture()
    vc.open(0)
    # Copied to my working folder from
    # /usr/local/Cellar/opencv/2.4.13.2/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml
    face_detector = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
    while vc.grab():
        (tf, im) = vc.read()
        im = cv2.resize(im, (im.shape[1] / 2, im.shape[0] / 2))  # Divide resolution by 2 for speed
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        k = cv2.waitKey(30)
        if k >= 0 and chr(k) == 'q':
            print "quitting"
            break

        # Detect faces
        face_boxes = face_detector.detectMultiScale(im_gray)
        for faceBox in face_boxes:
            classifySmile(im, im_gray, faceBox, w)
        cv2.imshow("WebCam", im)

    cv2.destroyWindow("WebCam")
    vc.release()


if __name__ == "__main__":
    # Load data
    # In ipython, use "run -i homework2_afguerrerohernan.py" to avoid re-loading of data
    if ('trainingFaces' not in globals()):
        trainingFaces = np.load("trainingFaces.npy")
        trainingLabels = np.load("trainingLabels.npy")
        testingFaces = np.load("testingFaces.npy")
        testingLabels = np.load("testingLabels.npy")

    w1 = method1(trainingFaces, trainingLabels, testingFaces, testingLabels)
    w2 = method2(trainingFaces, trainingLabels, testingFaces, testingLabels)
    w3 = method3(trainingFaces, trainingLabels, testingFaces, testingLabels)

    for w in [w1, w2, w3]:
        reportCosts(w, trainingFaces, trainingLabels, testingFaces, testingLabels)

        # detectSmiles(w3)  # Requires OpenCV
        # plot.show(plot.imshow(np.reshape(w3, (24, 24))))
