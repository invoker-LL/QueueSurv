# QueueSurv
This repo improves batch-size during Cox-loss computation in WSI survival prediction using memory queue.
For each iter, the final pooled feature (input of classfier head) of WSI is queued. During cox-loss computation, previous pooled features and hazards are computed with current one. 
This helps learn better risk ranking matrix.
