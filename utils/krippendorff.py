import sys
import nltk
from nltk.metrics import masi_distance
from nltk.metrics import agreement
from nltk.metrics.agreement import AnnotationTask


# takes as input two lists of annotations, labels per row, separated by whitespace

def read_labels(infile):
    labels = []
    file=open(infile)
    for line in file:
        ls = []
        line=line.strip().split(" ")
        for l in line:
            ls.append(l)
        ls=sorted(ls)
        labels.append(ls)
    return labels

annot_1 = read_labels(sys.argv[1])
annot_2 = read_labels(sys.argv[2])

tryme = []

def read_labels(annotator, lisst):
    annotator="annotator"+str(annotator)
    for i,annot in enumerate(lisst):#,annot_2_bin):
#        print("A", annot)
        i = str(i)
        myset = (annotator,i,frozenset(annot))
        tryme.append(myset)

#print("trme", tryme)
read_labels("1", annot_1)
#print("2", tryme)
read_labels("2", annot_2)
#print("3", tryme)

#task_data = [('coder1','Item0',frozenset(['l1','l2'])),
#('coder2','Item0',frozenset(['l1'])),
#('coder1','Item1',frozenset(['l1','l2'])),
#('coder2','Item1',frozenset(['l1','l2'])),
#('coder1','Item2',frozenset(['l1'])),
#('coder2','Item2',frozenset(['l1']))]

#print(krippendorff.alpha(reliability_data=data))
#toy_data = [('1', 5723, (1)),('2', 5723, (2))]
#task = AnnotationTask(data=[x.split() for x in open(os.path.join(os.path.dirname(__file__), "artstein_poesio_example.txt"))])

task = AnnotationTask(distance=masi_distance)
task.load_array(tryme)
print(task.alpha())
