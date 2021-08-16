# Neural Style Transfer

Pytorch implementation of the "Neural Algorithm of Artistic Style" by Leon Gatys et al

## Usage
---
```
python style_transfer.py --content [content image dir] --style [style image dir] --alpha [content weight] --beta [style weight] --gpu [use gpu] --output_size [output size] --output_file [output_file]
```

#### Optional parameters
- --alpha (default 1)
- --beta (default 1)
-  --output_file (default result.jpg)
-  --gpu (default False)
-  --output_size (default (200,200))