Discriminator(
  (mlps): ModuleList(
    (0): MLP(
      (linears): ModuleList(
        (0): Linear(in_features=5, out_features=64, bias=True)
        (1): Linear(in_features=64, out_features=64, bias=True)
      )
      (batch_norms): ModuleList(
        (0): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1-3): 3 x MLP(
      (linears): ModuleList(
        (0-1): 2 x Linear(in_features=64, out_features=64, bias=True)
      )
      (batch_norms): ModuleList(
        (0): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
  )
  (batch_norms): ModuleList(
    (0-3): 4 x BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (linears_prediction): ModuleList(
    (0): Linear(in_features=5, out_features=2, bias=True)
    (1-4): 4 x Linear(in_features=64, out_features=2, bias=True)
  )
)