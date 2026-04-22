# Final Project Code

Attribute-controlled facial generation with systematic accuracy evaluation.

## Project Overview

DenseNet169 backbone → BiLSTM over spatial rows → multi-label classifier

    Flow:
      Image (B,3,H,W)
        → DenseNet169 feature map  (B, 1664, h, w)   h=w=7 for 224px input
        → treat h rows as timesteps, each row = w*1664 features
        → BiLSTM(hidden=512, layers=2)
        → take last hidden state from both directions → (B, 1024)
        → Linear → (B, 40)

    Rationale for CelebA:
      Face images have a natural top-to-bottom spatial order
      (hair → forehead → eyes → nose → mouth → chin).
      BiLSTM can model both top-down and bottom-up dependencies
      across these spatial rows, which plain global-pooling discards.
