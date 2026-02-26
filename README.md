# Take Note: Your Molecular Dataset Is Probably Aligned
Did you know: Molecular datasets are typically generated with computational chemistry codes which do not randomize pose. Hence, the resulting geometries are usually not randomly oriented.
For example, if you plot the orientations of all molecules in the QM9 dataset in 2D, the result looks far from uniform:

<img src="https://github.com/user-attachments/assets/886b0942-222b-4746-9627-433dd13b8756" 
     width="1000"  />

Even more interesting: Structurally similar molecules are oriented similarly. 

Such orientation bias is easily overlooked but can have serious consequences. Check out our [ICLR26 paper](https://openreview.net/forum?id=zrCGvLOrTL) for more.  
This repo will soon contain several tools to analyze the orientations of molecules in your dataset.  
