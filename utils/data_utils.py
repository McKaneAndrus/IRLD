import os
import pickle as pkl
import uuid

def initialize_scopes(data_dir):
    q_scope = str(uuid.uuid4())
    dyn_scope = str(uuid.uuid4())

    with open(os.path.join(data_dir, 'q_scope.pkl'), 'wb') as f:
        pkl.dump(q_scope, f)

    with open(os.path.join(data_dir, 'dyn_scope.pkl'), 'wb') as f:
        pkl.dump(dyn_scope, f)

def load_scopes(data_dir):

    with open(os.path.join(data_dir, 'q_scope.pkl'), 'rb') as f:
        q_scope = pkl.load(f)

    with open(os.path.join(data_dir, 'dyn_scope.pkl'), 'rb') as f:
        dyn_scope = pkl.load(f)

    return q_scope, dyn_scope
