'''

Simpsons Face Detector

Silly script to go through a very large number of Simpsons stills and pick outs
images that contain no faces (these were then used to make afterspringfield -
http://afterspringfield.tumblr.com/tagged/afterspringfield/chron

The fun bit of this script is the small genetic algorithm used to fine-tune
human face detection settings to work better on simpson-like faces.

cv2 image detection code adapted from http://knowpapa.com/images-remote-directory-py/

@author: ajparsons
'''

import cv2
import os
import shutil
import random
import math


class FaceSetting(object):
    """
    Holds the settings for the facial detection.
    Can generate 'children' with random variation off current settings
    """

    def __init__(self,
                 scaleFactor=1.01293900138,
                 minNeighours=1,
                 minSize=20,
                 parent=None):

        if parent:
            self.scaleFactor = parent.scaleFactor
            self.minNeighbours = parent.minNeighbours
            self.minSize = parent.minSize
        else:
            self.scaleFactor = scaleFactor
            self.minNeighbours = minNeighours
            self.minSize = minSize

        self.score = 0
        self.face_score = 0
        self.noface_score = 0

    def output(self):
        """
        print settings and score
        """
        print "scaleFactor : {0}".format(self.scaleFactor)
        print "minNeighbours : {0}".format(self.minNeighbours)
        print "minSize : {0}".format(self.minSize)
        if self.score != 0:
            print "score : {0}".format(self.score)
            print "face score : {0}".format(self.face_score)
            print "no face score : {0}".format(self.noface_score)

    def variate(self, variation):
        """
        randomly mutates self according to given settings
        """
        properties = ["scaleFactor", "minNeighbours", "minSize"]
        for p in properties:
            current = float(getattr(self, p))
            upper = current + float(current) * variation
            lower = current - float(current) * variation

            """
            set floors and ceilings on mutation to avoid crashes
            """

            if p == "scaleFactor" and lower <= 1:
                lower = 1.01
            if p == "minSize" and lower <= 0:
                lower = 1
                if upper == 0:
                    upper = 4
            if p == "minNeighbours" and lower <= 1:
                lower = 1
                if upper == 0:
                    upper = 4

            new = random.uniform(lower, upper)
            if p in ["minNeighbours", "minSize"]:
                new = int(new)
            setattr(self, p, new)

    def random_children(self, count, variance=None):
        """
        generate random children
        if variation not set - decrease variance on
        entries with lower scores
        """

        children = []

        if variance is None:
            variance = math.sqrt(1.00 - self.score)

        for x in range(0, count):
            child = FaceSetting(parent=self)
            child.variate(variance)
            children.append(child)

        return children


def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)


def is_face(path, settings, display=False):
    """
    is there a face in the file that can be detected with the given settings?
    """
    global faceCascade
    image = cv2.imread(path)
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except:
        return True
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=settings.scaleFactor,
        minNeighbors=settings.minNeighbours,
        minSize=(settings.minSize, settings.minSize),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    if display and len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Faces found", image)
        cv2.waitKey(0)
    if len(faces) > 0:
        return True
    else:
        return False


def genetic_detect(next_pop=None, iteration=0):
    """
    Variate and iterate on the starting settings
    until we have a set of settings that reach the
    target accuracy rating
    """
    if iteration == 3:
        return None

    """
    if first iteration - we create some random
    children round the default settings.
    """
    if next_pop is None:
        starting = FaceSetting()
        next_pop = starting.random_children(7, 0.1)

    all_time = []
    best_score = 0
    target = 0.90
    generation = 0
    while best_score < target and generation < 50:
        score, next_pop = run_population(next_pop)
        generation += 1
        print "generation {0}".format(generation)
        best_score = score.score
        print "best score {0}".format(score.score)
        score.output()
        all_time.append(score)

    if best_score < target:
        """
        in failure - sort failures by score, take the top four scorers
        and use them to populate the next generation
        """
        print "target failed"
        all_time.sort(key=lambda x: x.score, reverse=True)
        random_children = []
        for a in all_time[:4]:
            random_children.extend(starting.random_children(7, 0.1))
        genetic_detect(random_children, iteration + 1)


def run_population(pop):
    """
    given training data divided into 'faces' and 'nofaces' folders,
    see how well the current settings do at matching the classification
    """

    face_folder = "E://projects//simpsons//face"
    no_face_folder = "E://projects//simpsons//noface"

    count = 0
    for r in pop:
        count += 1
        print "pop {0} of {1}".format(count, len(pop))
        r.output()
        print "getting face score"
        match, total = score_folder(face_folder, settings=r)
        r.face_score = float(match) / float(total)
        print "face score : {0}".format(r.face_score)
        print "getting non face score"
        nomatch, nototal = score_folder(
            no_face_folder, settings=r, reverse=True, weight=4)
        r.noface_score = float(nomatch) / float(nototal)
        print "nonface score : {0}".format(r.noface_score)
        r.score = (float(nomatch) + float(match)) / \
            (float(total) + float(nototal))
    pop.sort(key=lambda x: x.score, reverse=True)

    no_children = 8
    new_children = []
    for r in pop:
        new_children.extend(r.random_children(no_children))
        no_children = no_children / 2
        if no_children == 0:
            break

    best_score = pop[0]

    return best_score, new_children


def clear_face_folders(folder):
    """
    delete current contents of face folders for resorting
    from prime folder
    """
    subs = ["face", "noface"]
    folders = [os.path.join(folder, sub) for sub in subs]
    for fo in folders:
        filelist = [f for f in os.listdir(fo)]
        for f in filelist:
            os.remove(os.path.join(fo, f))


def score_folder(folder, settings, reverse=False, weight=1):
    """
    how many images in this folder have faces?
    """
    score = 0
    count = 0
    for f in os.listdir(folder):
        count += 1
        path = os.path.join(folder, f)
        if is_face(path, settings):
            score += 1

    score = score * weight
    count = count * weight

    if reverse:
        return count - score, count
    else:
        return score, count


def process_faces(folder):
    """
    now we have the optimal simpsons settings - process and sort
    """
    settings = FaceSetting()
    settings.scaleFactor = 1.01293900138
    settings.minNeighbours = 1
    settings.minSize = 20

    clear_face_folders(folder)
    for f in os.listdir(folder):
        print f
        if os.path.splitext(f)[1].lower() == ".jpg":
            path = os.path.join(folder, f)
            face_path = os.path.join(folder, "face", f)
            no_face_path = os.path.join(folder, "noface", f)
            if is_face(path, settings=settings):
                shutil.copyfile(path, face_path)
            else:
                shutil.copyfile(path, no_face_path)


def rename_files():
    root = "E:\\Projects\\simpsons\\"
    destination = "E:\\Projects\\simpsons_with\\"
    folders = [f for f in os.listdir(
        root) if os.path.isdir(os.path.join(root, f))]
    for folder in folders:
        path = os.path.join(root, folder)
        for f in os.listdir(path):
            if os.path.splitext(f)[1].lower() == ".jpg":
                dest = os.path.join(destination, "{0}_{1}".format(folder, f))
                print dest
                shutil.copy(os.path.join(path, f), dest)


if __name__ == "__main__":

    cascade_location = "E:\\programming\\data\\haarcascades\\haarcascade_frontalface_default.xml"
    face_location = "E://projects/simpsons_with/"

    faceCascade = cv2.CascadeClassifier(cascade_location)
    process_faces(face_location)
