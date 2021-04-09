import os
import pickle
from sklearn.decomposition import PCA
import numpy as np

SYMBOLS = ['(', '-', ')', 'n', '=', 'i', '\\sum', 'S', '1', 'r', '\\theta', '2',
           '\\pi', '8', 'd', 'a', '4', 'c', 'b', '+', '3', '7', 'y', '0', 'x',
           '\\sqrt', 'k', '\\sin', '\\cos', '\\pm', 'p', 'u', 'q', 'P', 'j',
           '\\leq', '|', '\\lt', '!', '\\ldots', 'f', 'C', 'B', 'A', 'R', 'm',
           'COMMA', 't', '\\prime', 'w', 's', '\\int', '\\alpha', 'H',
           '\\Delta', 'I', '\\in', 'M', 'X', '\\sigma', 'T', 'E', 'L',
           '\\forall', 'V', '\\mu', '\\{', '\\}', 'N', '\\neq', 'G', 'e', 'F',
           'g', '\\phi', '\\exists', '\\gt', 'z', '\\log', '9', '\\div', '5',
           '\\times', 'o', 'v', '6', '/', '\\gamma', '\\beta', 'h', '\\geq',
           'Y', '\\infty', 'l', '.', '\\rightarrow', '\\lim', '[', ']', '\\tan',
           '\\lambda']




def pca(ui2image, num_vectors, output_file, mode):
    imgs = []
    uis = []
    for ui in ui2image.keys():
        uis.append(ui)
        imgs.append(ui2image[ui])
    imgs = np.array(imgs)
    imgs = imgs.reshape((imgs.shape[0], -1))

    if not os.path.isfile(output_file):
        pca = PCA(num_vectors, svd_solver='randomized', whiten=True, random_state=16)
        # PCA on the dataset
        print('PCA: Fitting on dataset')
        pca.fit(imgs)
        with open(output_file, 'wb') as handler:
            pickle.dump(pca, handler, protocol=pickle.HIGHEST_PROTOCOL)
            print("PCA features dumped at " + output_file)
    else:
        with open(output_file, 'rb') as handler:
            pca = pickle.load(handler)
        print("PCA features loaded from " + output_file)

    print('PCA: Compressing dataset')
    compressed = pca.transform(imgs)

    ui2pca = {}
    for idx, ui in enumerate(uis):
        ui2pca[ui] = compressed[idx]
    return ui2pca


def predict(model, dataset, X=None):
    ui2preds = {}
    print("Generating predictions ...")
    if X is None:
        X = np.empty((len(dataset), dataset.num_pca_vectors))
        for i, ui in dataset.uis_dict.items():
            X[i] = dataset.pca_features[ui]
    probs = model.predict_proba(X)
    topk_classes = np.argsort(-probs, axis=1)[:, :10]
    for i, ui in dataset.uis_dict.items():
        ui2preds[ui] = [SYMBOLS[token] for token in topk_classes[i]]
    return ui2preds