% Clean grid has one RL cell and one POW cell, in different families,
% so drive the two existing family-specific runners and merge.
here='/Users/gino.diez/Documents/cbm_python/benchmark';
addpath(genpath(fullfile(here,'external','VBA-toolbox')));
addpath(fullfile(here,'matlab'));
data_dir=fullfile(here,'data','clean'); out_dir=fullfile(here,'results');

rows=struct('cell',{},'generator',{},'n_trials',{},'beta_cond',{}, ...
            'fitted_model',{},'subject',{},'est_alpha_pos',{}, ...
            'est_alpha_neg',{},'est_rho',{},'est_beta',{},'F',{}, ...
            'seconds',{},'converged',{});
t0=tic;
% ---------- RL cell (state-space) ----------
d=load(fullfile(data_dir,'RL.mat')); nsub=double(d.n_subjects); T=double(d.n_trials);
for mi=1:2
  if mi==1, mn='RL'; nth=1; else, mn='RL2'; nth=2; end
  tc=tic;
  for i=1:nsub
    choices=double(d.choices(i,:)); rewards=double(d.rewards(i,:));
    y=double(choices==0); u=zeros(2,T);
    u(1,2:T)=y(1:T-1); u(2,2:T)=rewards(1:T-1);
    dim=struct('n',2,'n_phi',1,'n_theta',nth);
    opt=struct(); opt.sources=struct('type',1,'out',1);
    opt.verbose=0; opt.DisplayWin=0;
    opt.priors.muX0=[0;0]; opt.priors.SigmaX0=zeros(2);
    opt.priors.a_alpha=Inf; opt.priors.b_alpha=0;
    opt.priors.muTheta=zeros(nth,1); opt.priors.SigmaTheta=10*eye(nth);
    opt.priors.muPhi=zeros(1,1);   opt.priors.SigmaPhi=10*eye(1);
    if mi==1, ff=@f_Qlearn; else, ff=@f_QlearnAsym2; end
    try
      [post,out]=VBA_NLStateSpaceModel(y,u,ff,@g_softmax,dim,opt);
      ap=VBA_sigmoid(post.muTheta(1));
      if mi==1, an=ap; else, an=VBA_sigmoid(post.muTheta(2)); end
      bet=exp(post.muPhi(1)); Fv=out.F(end); cv=1;
    catch ME
      fprintf('  RL %s subj %d FAILED: %s\n',mn,i,ME.message);
      ap=NaN; an=NaN; bet=NaN; Fv=NaN; cv=0;
    end
    rows(end+1)=struct('cell','RL','generator','RL','n_trials',T, ...
      'beta_cond',0,'fitted_model',mn,'subject',i-1,'est_alpha_pos',ap, ...
      'est_alpha_neg',an,'est_rho',NaN,'est_beta',bet,'F',Fv, ...
      'seconds',NaN,'converged',cv);
  end
  el=toc(tc); for k=(numel(rows)-nsub+1):numel(rows), rows(k).seconds=el/nsub; end
  fprintf('  RL  %-4s %6.2fs sum(F)=%10.2f\n',mn,el,sum([rows(end-nsub+1:end).F],'omitnan'));
end
% ---------- POW cell (static) ----------
d=load(fullfile(data_dir,'POW.mat')); nsub=double(d.n_subjects); T=double(d.n_trials);
for mi=1:2
  if mi==1, mn='LIN'; isLin=true; nphi=1; else, mn='POW'; isLin=false; nphi=2; end
  tc=tic;
  for i=1:nsub
    y=double(d.chose(i,:));
    u=[double(d.sure(i,:)); double(d.gamble(i,:)); double(d.prob(i,:))];
    dim=struct('n',0,'n_theta',0,'n_phi',nphi);
    opt=struct(); opt.sources=struct('type',1,'out',1);
    opt.verbose=0; opt.DisplayWin=0; opt.inG=struct('linear',isLin);
    opt.priors.muPhi=zeros(nphi,1); opt.priors.SigmaPhi=10*eye(nphi);
    try
      [post,out]=VBA_NLStateSpaceModel(y,u,[],@g_valuePower,dim,opt);
      if isLin, rh=1; bet=exp(post.muPhi(1));
      else, rh=exp(post.muPhi(1)); bet=exp(post.muPhi(2)); end
      Fv=out.F(end); cv=1;
    catch ME
      fprintf('  POW %s subj %d FAILED: %s\n',mn,i,ME.message);
      rh=NaN; bet=NaN; Fv=NaN; cv=0;
    end
    rows(end+1)=struct('cell','POW','generator','POW','n_trials',T, ...
      'beta_cond',0,'fitted_model',mn,'subject',i-1,'est_alpha_pos',NaN, ...
      'est_alpha_neg',NaN,'est_rho',rh,'est_beta',bet,'F',Fv, ...
      'seconds',NaN,'converged',cv);
  end
  el=toc(tc); for k=(numel(rows)-nsub+1):numel(rows), rows(k).seconds=el/nsub; end
  fprintf('  POW %-4s %6.2fs sum(F)=%10.2f\n',mn,el,sum([rows(end-nsub+1:end).F],'omitnan'));
end
save(fullfile(out_dir,'vba_clean.mat'),'rows','-v7');
fprintf('\n%d rows -> vba_clean.mat (%.1fs)\n',numel(rows),toc(t0));
