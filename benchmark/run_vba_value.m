% MATLAB VBA arm for the VALUE-FUNCTION grids (LIN vs POW risky choice).
%
% Reads the same .mat datasets the Python arms read and inverts each
% subject under both candidate models, writing
% benchmark/results/vba_<grid>.mat for benchmark/analyze.py.
%
% Usage:
%   matlab -batch "grid='value_recovery'; run('benchmark/run_vba_value.m')"
%
% These are STATIC models — no hidden state, so VBA is called with
% dim.n = 0 and an empty evolution function. That makes this arm a
% cleaner comparison than the RL one: with no state-space machinery in
% play, VBA and CBM differ only in how they approximate the posterior
% (variational vs Laplace), not in what model they fit.
%
% Comparability notes carried over from run_vba_arm.m: VBA's out.F is a
% variational free energy, CBM's log_evidence a Laplace approximation —
% comparable in kind, not in value, so model SELECTION is the fair
% comparison. Priors are matched to CBM's N(0,10) on every parameter.

if ~exist('grid', 'var'); grid = 'value_recovery'; end

here     = fileparts(mfilename('fullpath'));
vba_path = fullfile(here, 'external', 'VBA-toolbox');
data_dir = fullfile(here, 'data', grid);
out_dir  = fullfile(here, 'results');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end

addpath(genpath(vba_path));
addpath(fullfile(here, 'matlab'));

manifest = jsondecode(fileread(fullfile(data_dir, 'manifest.json')));
fprintf('VBA value arm: %d cells from %s\n', numel(manifest), data_dir);

rows = struct('cell', {}, 'generator', {}, 'n_trials', {}, ...
              'beta_cond', {}, 'fitted_model', {}, 'subject', {}, ...
              'est_rho', {}, 'est_beta', {}, 'F', {}, 'seconds', {}, ...
              'converged', {});

t_start = tic;
for ci = 1:numel(manifest)
    name = manifest(ci).name;
    d    = load(fullfile(data_dir, [name '.mat']));
    nsub = double(d.n_subjects);
    T    = double(d.n_trials);

    for mi = 1:2
        if mi == 1; model_name = 'LIN'; else; model_name = 'POW'; end
        isLin  = strcmp(model_name, 'LIN');
        t_cell = tic;
        for i = 1:nsub
            y = double(d.chose(i, :));              % 1 = chose gamble
            u = [double(d.sure(i, :));              % sure amount
                 double(d.gamble(i, :));            % gamble amount
                 double(d.prob(i, :))];             % gamble probability

            dim = struct('n', 0, 'n_theta', 0, 'n_phi', 2);
            if isLin; dim.n_phi = 1; end

            opt = struct();
            opt.sources    = struct('type', 1, 'out', 1);   % binomial
            opt.verbose    = 0;
            opt.DisplayWin = 0;
            opt.inG        = struct('linear', isLin);
            % Match CBM's prior exactly: N(0, 10) per parameter
            opt.priors.muPhi    = zeros(dim.n_phi, 1);
            opt.priors.SigmaPhi = 10 * eye(dim.n_phi);

            try
                [post, out] = VBA_NLStateSpaceModel(y, u, [], ...
                                                    @g_valuePower, dim, opt);
                if isLin
                    rho = 1.0;  bet = exp(post.muPhi(1));
                else
                    rho = exp(post.muPhi(1));  bet = exp(post.muPhi(2));
                end
                Fval = out.F(end);  conv = 1;
            catch ME
                fprintf('  %s %s subj %d FAILED: %s\n', name, model_name, i, ME.message);
                rho = NaN; bet = NaN; Fval = NaN; conv = 0;
            end

            rows(end+1) = struct( ...
                'cell', name, 'generator', d.generator, ...
                'n_trials', T, 'beta_cond', double(d.beta), ...
                'fitted_model', model_name, 'subject', i - 1, ...
                'est_rho', rho, 'est_beta', bet, 'F', Fval, ...
                'seconds', NaN, 'converged', conv);  %#ok<SAGROW>
        end
        el = toc(t_cell);
        for k = (numel(rows) - nsub + 1):numel(rows)
            rows(k).seconds = el / nsub;
        end
        fprintf('  [%d/%d] %-24s %-4s %6.2fs  sum(F)=%10.2f\n', ...
                ci, numel(manifest), name, model_name, el, ...
                sum([rows(end-nsub+1:end).F], 'omitnan'));
    end
end

outfile = fullfile(out_dir, ['vba_' grid '.mat']);
save(outfile, 'rows', '-v7');
fprintf('\n%d rows -> %s  (%.1fs total)\n', numel(rows), outfile, toc(t_start));
