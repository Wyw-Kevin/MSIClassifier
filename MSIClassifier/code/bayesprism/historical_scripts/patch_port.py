from pathlib import Path
import shutil,hashlib,json
P=Path(__file__).parent;V=P/'vendor/pybayesprism'; orig=P/'vendor_original/pybayesprism'
if not orig.exists():shutil.copytree(V,orig)
def edit(name,pairs):
 p=V/name;s=p.read_text()
 for a,b in pairs:
  assert a in s,(name,a);s=s.replace(a,b)
 p.write_text(s)
edit('gibbs.py',[
 ('for i in range(np.max(gibbs_idx)):','for i in range(int(np.max(gibbs_idx)) + 1):'),
 ('if seed is not None:\n            np.random.seed(seed)','rng = np.random.default_rng(seed) if not isinstance(seed, dict) else np.random.Generator(np.random.PCG64())\n        if isinstance(seed, dict):\n            rng.bit_generator.state = seed'),
 ('for g in range(G):\n                pvals = prob_mat[:, g] / np.sum(prob_mat[:, g])\n                Z_n_i[g, :] = np.random.multinomial(n = X_n[g], pvals = pvals)','pvals = (prob_mat / prob_mat.sum(axis=0)).T\n            Z_n_i = rng.multinomial(np.asarray(X_n, dtype=np.int64), pvals)'),
 ('GibbsSampler.rdirichlet(alpha = Z_nk_i + alpha)','rng.dirichlet(Z_nk_i + alpha)'),
 ('GibbsSampler.rdirichlet(alpha = np.sum(Z_n_i, axis = 0) + alpha)','rng.dirichlet(np.sum(Z_n_i, axis=0) + alpha)'),
 ('theta_post.ThetaPost.new(self.X.index, self.X.columns, gibbs_list)','theta_post.ThetaPost.new(self.X.index, phi.index, gibbs_list)')])
edit('process_input.py',[
 ('gene_df.loc[gene_df.iloc[:,0]==i, 1]','gene_df.loc[gene_df.iloc[:,0]==i].iloc[:, 1]'),
 ('input_filtered = input.loc[:, selected_gene_idx.tolist()]','selected_names = set(gene_df.loc[selected_gene_idx, "gene_name"])\n    input_filtered = input.loc[:, [g in selected_names for g in input_genes]]'),
 ('np.any(np.isinf(input_ndarray))','np.any(~np.isfinite(input_ndarray))')])
edit('optim.py',[
 ('value =  [log_posterior_gamma(x, args = args) for x in res.x]','value = log_posterior_gamma(res.x, args=args)'),
 ('opt_control.pop("sigma")','opt_control = opt_control.copy()\n    opt_control.pop("sigma")')])
edit('prism.py',[("'seed': 123","'seed': 2026")])
h={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for root in [V,orig] for p in root.rglob('*.py')}
(P/'vendor_source_hashes.json').write_text(json.dumps(h,indent=2))
