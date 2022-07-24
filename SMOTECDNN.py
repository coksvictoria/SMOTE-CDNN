import numpy as np ##numpy for sorting distance
import scipy##for distance metrics

from imblearn.base import BaseSampler
from sklearn.utils import _safe_indexing

from imblearn.over_sampling import SMOTE
from sklearn.base import clone

class EditedCDNN(BaseSampler):

    def __init__(self,*,sampling_strategy="auto", n_neighbors=5,kind_sel="cd",n_jobs=None,sampling_type="under-sampling"):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.kind_sel = kind_sel
        self.n_jobs = n_jobs
        self.sampling_strategy=sampling_strategy
        self._sampling_type=sampling_type
        self.sampling_type=sampling_type

        SAMPLING_TARGET_KIND=["minority","majority", "not minority","not majority","all","auto"]
        
        if isinstance(sampling_strategy, str):
           if sampling_strategy not in SAMPLING_TARGET_KIND:
            raise ValueError(
                f"When 'sampling_strategy' is a string, it needs"
                f" to be one of {SAMPLING_TARGET_KIND}. Got '{sampling_strategy}' "
                f"instead.")    

    def _fit_resample(self, X, y):
          #calculate distance
        d=scipy.spatial.distance.cdist(X,X)
        #get k lowest distance and save to Sx
        indexes=np.argsort(d)[:,1:self.n_neighbors+1] # return k indexes of lowest value in d

        idx_under = np.empty((0,), dtype=int)
        input_dim=X.shape[1]

        if self.kind_sel=="all":
          idx_under=np.flatnonzero(np.max(y[indexes],axis=1) == np.min(y[indexes],axis=1))
        elif self.kind_sel=="cd":
          y_pred=[] ##set y_predict list
          for n,index in enumerate(indexes): ##looping through k indexes over the whole test dataset
            Sx = dict()
            for idx in range(self.n_neighbors):
              key = index[idx]
              if y[key] in Sx:
                Sx[y[key]].append(X[key])
              else:
                Sx[y[key]] = []
                Sx[y[key]].append(X[key])

            #calculate current centroids within training dataset
            px = dict()
            for key in Sx:
              sum_item = np.zeros(input_dim)
              for i in range(len(Sx[key])):
                sum_item += Sx[key][i]

              px_item = sum_item/len(Sx[key])

              px[key] = px_item

            #calculate new centroid by adding new test data
            qx = dict()
            for key in Sx:
              sum_item = np.zeros(input_dim)
              for i in range(len(Sx[key])):
                sum_item+=Sx[key][i]
              sum_item += X[n]
              qx_item = sum_item/(len(Sx[key]) + 1)
              qx[key] = qx_item

            #calculate displacement
            theta = dict()
            for key in px:
              if key in qx:
                theta[key] = np.linalg.norm(px[key] - qx[key])

            label=min(theta, key=theta.get)
            y_pred.append(label)

          idx_under=np.flatnonzero(np.array(y_pred)==y)

        minority_class=np.argmin(np.bincount(y))
        majority_class=np.array(np.argmax(np.bincount(y)))
        all_class=np.unique(y)
        non_minority=np.setdiff1d(all_class,minority_class)
        non_majority=np.setdiff1d(all_class,majority_class)

        if self.sampling_strategy in ['not minority','auto']:
          target_class_indices= np.flatnonzero(y == int(minority_class))
          idx_under = np.unique(np.concatenate((idx_under,target_class_indices),axis=0))
        elif self.sampling_strategy=='not majority':
          target_class_indices= np.flatnonzero(y == majority_class)
          idx_under = np.unique(np.concatenate((idx_under,target_class_indices),axis=0))
        elif self.sampling_strategy=='all':
          pass
        elif self.sampling_strategy=='majority':
          for target_class in non_majority:
            target_class_indices= np.flatnonzero(y == target_class)
            idx_under = np.unique(np.concatenate((idx_under,target_class_indices),axis=0))
        elif self.sampling_strategy=='minority':
          for target_class in non_minority:
            target_class_indices= np.flatnonzero(y == target_class)
            idx_under = np.unique(np.concatenate((idx_under,target_class_indices),axis=0))

        self.sample_indices_ = idx_under

        return _safe_indexing(X, idx_under), _safe_indexing(y, idx_under)

    def _more_tags(self):
        return {"sample_indices": True}



class SMOTECDNN(BaseSampler):

    _sampling_type = "under-sampling"
    def __init__(self,sampling_strategy="auto",random_state=None,smote=None,cdnn=None,n_jobs=None):
        super().__init__()
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.smote = smote
        self.cdnn = cdnn
        self.n_jobs = n_jobs

    def _validate_estimator(self):
        "Private function to validate SMOTE and ECDNN objects"
        if self.smote is not None:
            if isinstance(self.smote, SMOTE):
                self.smote_ = clone(self.smote)
            else:
                raise ValueError(
                    f"smote needs to be a SMOTE object."
                    f"Got {type(self.smote)} instead."
                )
        # Otherwise create a default SMOTE
        else:
            self.smote_ = SMOTE(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )

        if self.cdnn is not None:
            if isinstance(self.cdnn, EditedCDNN):
                self.cdnn_ = clone(self.cdnn)
            else:
                raise ValueError(
                    f"enn needs to be an EditedCDNN."
                    f" Got {type(self.cdnn)} instead."
                )
        # Otherwise create a default ECDNN
        else:
            self.cdnn_ =  EditedCDNN(
                sampling_strategy="all", n_jobs=self.n_jobs)

    def _fit_resample(self, X, y):
        self._validate_estimator()
        self.sampling_strategy_ = self.sampling_strategy

        X_res, y_res = self.smote_.fit_resample(X, y)
        return self.cdnn_.fit_resample(X_res, y_res)