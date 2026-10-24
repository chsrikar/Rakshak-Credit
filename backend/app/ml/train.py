import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, f1_score, precision_score, recall_score, accuracy_score, brier_score_loss, confusion_matrix
from ..features import FEATURES, TARGET

def metrics(y, p, threshold):
    pred=(p>=threshold).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {"roc_auc":float(roc_auc_score(y,p)),"pr_auc_average_precision":float(average_precision_score(y,p)),"threshold":float(threshold),"accuracy":float(accuracy_score(y,pred)),"precision":float(precision_score(y,pred,zero_division=0)),"recall":float(recall_score(y,pred,zero_division=0)),"f1":float(f1_score(y,pred,zero_division=0)),"brier_score":float(brier_score_loss(y,p)),"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)}

def choose_threshold(y,p):
    precision,recall,thresholds=precision_recall_curve(y,p)
    if not len(thresholds): return .5
    f1=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-12)
    return float(thresholds[int(np.nanargmax(f1))])

def load_split(path):
    df=pd.read_csv(path); missing=[c for c in FEATURES+[TARGET] if c not in df.columns]
    if missing: raise ValueError(f"{path} missing columns: {missing}")
    return df[FEATURES].replace([np.inf,-np.inf],np.nan).fillna(0), df[TARGET].astype(int)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--data-dir',default='data'); parser.add_argument('--out-dir',default='artifacts'); args=parser.parse_args()
    data=Path(args.data_dir); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    Xtr,ytr=load_split(data/'train.csv'); Xv,yv=load_split(data/'validation.csv'); Xt,yt=load_split(data/'test.csv')
    if ytr.nunique()!=2: raise ValueError('Training target must contain both classes.')
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    grid=GridSearchCV(GradientBoostingClassifier(random_state=42),
      {'n_estimators':[50,100,150], 'learning_rate':[.03,.05,.1], 'max_depth':[1,2,3], 'min_samples_leaf':[5,10,20], 'subsample':[.8,1.0]},
      scoring='roc_auc',cv=cv,n_jobs=-1,refit=True)
    print('Tuning Gradient Boosting with 5-fold CV on training split only...')
    grid.fit(Xtr,ytr); gb=grid.best_estimator_; valp=gb.predict_proba(Xv)[:,1]; threshold=choose_threshold(yv,valp)
    print('Training Logistic Regression benchmark...')
    lr=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,class_weight='balanced',random_state=42)); lr.fit(Xtr,ytr)
    lr_valp=lr.predict_proba(Xv)[:,1]; lr_threshold=choose_threshold(yv,lr_valp)
    rows=[]
    for name,model,t in [('GradientBoosting_tuned',gb,threshold),('LogisticRegression',lr,lr_threshold)]:
        prob=model.predict_proba(Xt)[:,1]; result=metrics(yt,prob,t); result['model']=name; rows.append(result)
        print(f"{name}: ROC-AUC={result['roc_auc']:.3f} PR-AUC={result['pr_auc_average_precision']:.3f} F1={result['f1']:.3f} recall={result['recall']:.3f}")
    pd.DataFrame(rows).to_csv(out/'test_model_comparison.csv',index=False)
    dump({'model':gb,'model_name':'Tuned Gradient Boosting','features':FEATURES,'synthetic_only':True,'threshold_from_validation':threshold},out/'best_model.joblib')
    with open(out/'training_report.json','w',encoding='utf-8') as f: json.dump({'synthetic_only':True,'target':TARGET,'features':FEATURES,'train_rows':len(Xtr),'validation_rows':len(Xv),'test_rows':len(Xt),'cv_best_roc_auc':float(grid.best_score_),'best_params':grid.best_params_,'threshold_from_validation':threshold,'models':rows,'warning':'Metrics describe agreement with the supplied target only. If target is synthetic, this is not real repayment-risk validation.'},f,indent=2)
    print(f"Saved model and report to {out.resolve()}")
if __name__=='__main__': main()
