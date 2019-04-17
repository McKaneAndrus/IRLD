import os
import pickle as pkl
import uuid

def initialize_data_files(data_dir):
    im_scope = str(uuid.uuid4())
    q_scope = str(uuid.uuid4())
    invsas_scope = str(uuid.uuid4())
    invadt_scope = str(uuid.uuid4())

    with open(os.path.join(data_dir, 'im_scope.pkl'), 'wb') as f:
        pkl.dump(im_scope, f)

    with open(os.path.join(data_dir, 'q_scope.pkl'), 'wb') as f:
        pkl.dump(q_scope, f)

    with open(os.path.join(data_dir, 'invsas_scope.pkl'), 'wb') as f:
        pkl.dump(invsas_scope, f)

    with open(os.path.join(data_dir, 'invadt_scope.pkl'), 'wb') as f:
        pkl.dump(invadt_scope, f)

def load_data(data_dir):
    with open(os.path.join(data_dir, 'im_scope.pkl'), 'rb') as f:
        im_scope = pkl.load(f)

    with open(os.path.join(data_dir, 'q_scope.pkl'), 'rb') as f:
        q_scope = pkl.load(f)

    with open(os.path.join(data_dir, 'invsas_scope.pkl'), 'rb') as f:
        invsas_scope = pkl.load(f)

    with open(os.path.join(data_dir, 'invadt_scope.pkl'), 'rb') as f:
        invadt_scope = pkl.load(f)

    return im_scope, q_scope, invsas_scope, invadt_scope
