(multivec) dhy@ta:~/multi-vector-retrieval$ python3 script/data/build_index.py
plaid start lotte-500-gnd
# gpu 4
rm -rf /home/dhy/Dataset/multi-vector-retrieval/Embedding/lotte-500-gnd
ColBERTConfig(query_token_id='[unused0]', doc_token_id='[unused1]', query_token='[Q]', doc_token='[D]', ncells=None, centroid_score_threshold=None, ndocs=None, index_path=None, nbits=2, kmeans_niters=4, resume=False, similarity='cosine', bsize=32, accumsteps=1, lr=3e-06, maxsteps=500000, save_every=None, warmup=None, warmup_bert=None, relu=False, nway=2, use_ib_negatives=False, reranker=False, distillation_alpha=1.0, ignore_scores=False, model_name='bert-base-uncased', query_maxlen=32, attend_to_mask_tokens=False, interaction='colbert', dim=128, doc_maxlen=220, mask_punctuation=True, checkpoint=None, triples=None, collection=None, queries=None, index_name=None, overwrite=False, root='/home/dhy/multi-vector-retrieval/baseline/ColBERT', experiment='default', index_root=None, name='2025-04/02/15.26.15', rank=0, nranks=1, amp=True, gpus=4)


[Apr 02, 15:26:22] #> Note: Output directory /home/dhy/multi-vector-retrieval/baseline/ColBERT/experiments/lotte-500-gnd/indexes/lotte-500-gnd already exists


[Apr 02, 15:26:22] #> Will delete 1 files already at /home/dhy/multi-vector-retrieval/baseline/ColBERT/experiments/lotte-500-gnd/indexes/lotte-500-gnd in 20 seconds...
#> Starting...
#> Starting...
#> Starting...
#> Starting...
nranks = 4       num_gpus = 4    device=2
nranks = 4       num_gpus = 4    device=3
nranks = 4       num_gpus = 4    device=1
nranks = 4       num_gpus = 4    device=0
[W402 15:26:58.037987507 Utils.hpp:164] Warning: Environment variable NCCL_BLOCKING_WAIT is deprecated; use TORCH_NCCL_BLOCKING_WAIT instead (function operator())
[Apr 02, 15:26:58] #> Loading collection...
0M 
[W402 15:26:58.156322878 Utils.hpp:164] Warning: Environment variable NCCL_BLOCKING_WAIT is deprecated; use TORCH_NCCL_BLOCKING_WAIT instead (function operator())
[Apr 02, 15:26:58] #> Loading collection...
0M 
[W402 15:26:58.765155445 Utils.hpp:164] Warning: Environment variable NCCL_BLOCKING_WAIT is deprecated; use TORCH_NCCL_BLOCKING_WAIT instead (function operator())
[Apr 02, 15:26:58] #> Loading collection...
0M [W402 15:26:58.768642016 Utils.hpp:164] Warning: Environment variable NCCL_BLOCKING_WAIT is deprecated; use TORCH_NCCL_BLOCKING_WAIT instead (function operator())
{
    "query_token_id": "[unused0]",
    "doc_token_id": "[unused1]",
    "query_token": "[Q]",
    "doc_token": "[D]",
    "ncells": null,
    "centroid_score_threshold": null,
    "ndocs": null,
    "index_path": null,
    "nbits": 2,
    "kmeans_niters": 20,
    "resume": false,
    "similarity": "cosine",
    "bsize": 64,
    "accumsteps": 1,
    "lr": 1e-5,
    "maxsteps": 400000,
    "save_every": null,
    "warmup": 20000,
    "warmup_bert": null,
    "relu": false,
    "nway": 64,
    "use_ib_negatives": true,
    "reranker": false,
    "distillation_alpha": 1.0,
    "ignore_scores": false,
    "model_name": "bert-base-uncased",
    "query_maxlen": 32,
    "attend_to_mask_tokens": false,
    "interaction": "colbert",
    "dim": 128,
    "doc_maxlen": 180,
    "mask_punctuation": true,
    "checkpoint": "\/home\/dhy\/Dataset\/multi-vector-retrieval\/RawData\/colbert-pretrain\/colbertv2.0",
    "triples": "\/future\/u\/okhattab\/root\/unit\/experiments\/2021.10\/downstream.distillation.round2.2_score\/round2.nway6.cosine.ib\/examples.64.json",
    "collection": "\/home\/dhy\/Dataset\/multi-vector-retrieval\/RawData\/lotte-500-gnd\/document\/collection.tsv",
    "queries": "\/future\/u\/okhattab\/data\/MSMARCO\/queries.train.tsv",
    "index_name": "lotte-500-gnd",
    "overwrite": false,
    "root": "\/home\/dhy\/multi-vector-retrieval\/baseline\/ColBERT\/experiments",
    "experiment": "lotte-500-gnd",
    "index_root": null,
    "name": "2025-04\/02\/15.26.15",
    "rank": 0,
    "nranks": 4,
    "amp": true,
    "gpus": 4
}
[Apr 02, 15:26:58] #> Loading collection...
0M 

[Apr 02, 15:26:59] Loading segmented_maxsim_cpp extension (set COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True for more info)...
[Apr 02, 15:26:59] Loading segmented_maxsim_cpp extension (set COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True for more info)...
/home/dhy/multi-vector-retrieval/script/data/../../baseline/ColBERT/colbert/utils/amp.py:12: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  self.scaler = torch.cuda.amp.GradScaler()
/home/dhy/multi-vector-retrieval/script/data/../../baseline/ColBERT/colbert/utils/amp.py:12: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  self.scaler = torch.cuda.amp.GradScaler()
[Apr 02, 15:27:00] Loading segmented_maxsim_cpp extension (set COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True for more info)...
[Apr 02, 15:27:00] Loading segmented_maxsim_cpp extension (set COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True for more info)...
[Apr 02, 15:27:00] [1]           #> Encoding 126 passages..
/home/dhy/multi-vector-retrieval/script/data/../../baseline/ColBERT/colbert/utils/amp.py:15: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  return torch.cuda.amp.autocast() if self.activated else NullContextManager()
/home/dhy/multi-vector-retrieval/script/data/../../baseline/ColBERT/colbert/utils/amp.py:12: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  self.scaler = torch.cuda.amp.GradScaler()
/home/dhy/multi-vector-retrieval/script/data/../../baseline/ColBERT/colbert/utils/amp.py:12: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  self.scaler = torch.cuda.amp.GradScaler()
[Apr 02, 15:27:01] [3]           #> Encoding 122 passages..
/home/dhy/multi-vector-retrieval/script/data/../../baseline/ColBERT/colbert/utils/amp.py:15: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  return torch.cuda.amp.autocast() if self.activated else NullContextManager()
[Apr 02, 15:27:02] [2]           #> Encoding 126 passages..
/home/dhy/multi-vector-retrieval/script/data/../../baseline/ColBERT/colbert/utils/amp.py:15: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  return torch.cuda.amp.autocast() if self.activated else NullContextManager()
[Apr 02, 15:27:02] [0]           # of sampled PIDs = 500         sampled_pids[:3] = [213, 375, 5]
[Apr 02, 15:27:02] [0]           #> Encoding 126 passages..
/home/dhy/multi-vector-retrieval/script/data/../../baseline/ColBERT/colbert/utils/amp.py:15: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  return torch.cuda.amp.autocast() if self.activated else NullContextManager()
ta:963210:963210 [0] NCCL INFO Bootstrap : Using enp233s0f0:10.120.17.132<0>
ta:963210:963210 [0] NCCL INFO NET/Plugin : dlerror=libnccl-net.so: cannot open shared object file: No such file or directory No plugin found (libnccl-net.so), using internal implementation

ta:963210:963210 [0] misc/cudawrap.cc:188 NCCL WARN Failed to find CUDA library libcuda.so (NCCL_CUDA_PATH='') : libcuda.so: cannot open shared object file: No such file or directory
NCCL version 2.20.5+cuda12.4

ta:963212:963212 [2] misc/cudawrap.cc:188 NCCL WARN Failed to find CUDA library libcuda.so (NCCL_CUDA_PATH='') : �a?S�U
ta:963212:963212 [2] NCCL INFO Bootstrap : Using enp233s0f0:10.120.17.132<0>
ta:963212:963212 [2] NCCL INFO NET/Plugin : dlerror=libnccl-net.so: cannot open shared object file: No such file or directory No plugin found (libnccl-net.so), using internal implementation

ta:963214:963214 [3] misc/cudawrap.cc:188 NCCL WARN Failed to find CUDA library libcuda.so (NCCL_CUDA_PATH='') : �z�!IV
ta:963214:963214 [3] NCCL INFO Bootstrap : Using enp233s0f0:10.120.17.132<0>

ta:963211:963211 [1] misc/cudawrap.cc:188 NCCL WARN Failed to find CUDA library libcuda.so (NCCL_CUDA_PATH='') :  ����U
ta:963214:963214 [3] NCCL INFO NET/Plugin : dlerror=libnccl-net.so: cannot open shared object file: No such file or directory No plugin found (libnccl-net.so), using internal implementation
ta:963211:963211 [1] NCCL INFO Bootstrap : Using enp233s0f0:10.120.17.132<0>
ta:963211:963211 [1] NCCL INFO NET/Plugin : dlerror=libnccl-net.so: cannot open shared object file: No such file or directory No plugin found (libnccl-net.so), using internal implementation
ta:963214:964403 [3] NCCL INFO NET/IB : No device found.
ta:963214:964403 [3] NCCL INFO NET/Socket : Using [0]enp233s0f0:10.120.17.132<0>
ta:963214:964403 [3] NCCL INFO Using non-device net plugin version 0
ta:963214:964403 [3] NCCL INFO Using network Socket
ta:963210:964401 [0] NCCL INFO NET/IB : No device found.
ta:963210:964401 [0] NCCL INFO NET/Socket : Using [0]enp233s0f0:10.120.17.132<0>
ta:963210:964401 [0] NCCL INFO Using non-device net plugin version 0
ta:963210:964401 [0] NCCL INFO Using network Socket
ta:963211:964404 [1] NCCL INFO NET/IB : No device found.
ta:963211:964404 [1] NCCL INFO NET/Socket : Using [0]enp233s0f0:10.120.17.132<0>
ta:963211:964404 [1] NCCL INFO Using non-device net plugin version 0
ta:963211:964404 [1] NCCL INFO Using network Socket
ta:963212:964402 [2] NCCL INFO NET/IB : No device found.
ta:963212:964402 [2] NCCL INFO NET/Socket : Using [0]enp233s0f0:10.120.17.132<0>
ta:963212:964402 [2] NCCL INFO Using non-device net plugin version 0
ta:963212:964402 [2] NCCL INFO Using network Socket
ta:963214:964403 [3] NCCL INFO comm 0x564925139fa0 rank 3 nranks 4 cudaDev 3 nvmlDev 3 busId c7000 commId 0x425d648351493254 - Init START
ta:963211:964404 [1] NCCL INFO comm 0x55f6974375f0 rank 1 nranks 4 cudaDev 1 nvmlDev 1 busId 67000 commId 0x425d648351493254 - Init START
ta:963210:964401 [0] NCCL INFO comm 0x55d838843b50 rank 0 nranks 4 cudaDev 0 nvmlDev 0 busId 25000 commId 0x425d648351493254 - Init START
ta:963212:964402 [2] NCCL INFO comm 0x55f957686b20 rank 2 nranks 4 cudaDev 2 nvmlDev 2 busId 87000 commId 0x425d648351493254 - Init START
ta:963210:964401 [0] NCCL INFO Setting affinity for GPU 0 to ffffffff,00000000,ffffffff
ta:963214:964403 [3] NCCL INFO Setting affinity for GPU 3 to ffffffff,00000000,ffffffff,00000000
ta:963211:964404 [1] NCCL INFO Setting affinity for GPU 1 to ffffffff,00000000,ffffffff
ta:963212:964402 [2] NCCL INFO Setting affinity for GPU 2 to ffffffff,00000000,ffffffff,00000000
ta:963214:964403 [3] NCCL INFO comm 0x564925139fa0 rank 3 nRanks 4 nNodes 1 localRanks 4 localRank 3 MNNVL 0
ta:963212:964402 [2] NCCL INFO comm 0x55f957686b20 rank 2 nRanks 4 nNodes 1 localRanks 4 localRank 2 MNNVL 0
ta:963214:964403 [3] NCCL INFO Trees [0] -1/-1/-1->3->2 [1] -1/-1/-1->3->2
ta:963214:964403 [3] NCCL INFO P2P Chunksize set to 131072
ta:963212:964402 [2] NCCL INFO Trees [0] 3/-1/-1->2->1 [1] 3/-1/-1->2->1
ta:963212:964402 [2] NCCL INFO P2P Chunksize set to 131072
ta:963211:964404 [1] NCCL INFO comm 0x55f6974375f0 rank 1 nRanks 4 nNodes 1 localRanks 4 localRank 1 MNNVL 0
ta:963210:964401 [0] NCCL INFO comm 0x55d838843b50 rank 0 nRanks 4 nNodes 1 localRanks 4 localRank 0 MNNVL 0
ta:963211:964404 [1] NCCL INFO Trees [0] 2/-1/-1->1->0 [1] 2/-1/-1->1->0
ta:963211:964404 [1] NCCL INFO P2P Chunksize set to 131072
ta:963210:964401 [0] NCCL INFO Channel 00/02 :    0   1   2   3
ta:963210:964401 [0] NCCL INFO Channel 01/02 :    0   1   2   3
ta:963210:964401 [0] NCCL INFO Trees [0] 1/-1/-1->0->-1 [1] 1/-1/-1->0->-1
ta:963210:964401 [0] NCCL INFO P2P Chunksize set to 131072
ta:963212:964402 [2] NCCL INFO Channel 00/0 : 2[2] -> 3[3] via P2P/IPC
ta:963214:964403 [3] NCCL INFO Channel 00/0 : 3[3] -> 0[0] via P2P/IPC
ta:963211:964404 [1] NCCL INFO Channel 00/0 : 1[1] -> 2[2] via P2P/IPC
ta:963214:964403 [3] NCCL INFO Channel 01/0 : 3[3] -> 0[0] via P2P/IPC
ta:963212:964402 [2] NCCL INFO Channel 01/0 : 2[2] -> 3[3] via P2P/IPC
ta:963210:964401 [0] NCCL INFO Channel 00/0 : 0[0] -> 1[1] via P2P/IPC
ta:963211:964404 [1] NCCL INFO Channel 01/0 : 1[1] -> 2[2] via P2P/IPC
ta:963210:964401 [0] NCCL INFO Channel 01/0 : 0[0] -> 1[1] via P2P/IPC
ta:963212:964402 [2] NCCL INFO Connected all rings
ta:963212:964402 [2] NCCL INFO Channel 00/0 : 2[2] -> 1[1] via P2P/IPC
ta:963211:964404 [1] NCCL INFO Connected all rings
ta:963214:964403 [3] NCCL INFO Connected all rings
ta:963210:964401 [0] NCCL INFO Connected all rings
ta:963214:964403 [3] NCCL INFO Channel 00/0 : 3[3] -> 2[2] via P2P/IPC
ta:963212:964402 [2] NCCL INFO Channel 01/0 : 2[2] -> 1[1] via P2P/IPC
ta:963214:964403 [3] NCCL INFO Channel 01/0 : 3[3] -> 2[2] via P2P/IPC
ta:963211:964404 [1] NCCL INFO Channel 00/0 : 1[1] -> 0[0] via P2P/IPC
ta:963211:964404 [1] NCCL INFO Channel 01/0 : 1[1] -> 0[0] via P2P/IPC
ta:963210:964401 [0] NCCL INFO Connected all trees
ta:963214:964403 [3] NCCL INFO Connected all trees
ta:963214:964403 [3] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 512 | 512
ta:963214:964403 [3] NCCL INFO 2 coll channels, 0 collnet channels, 0 nvls channels, 2 p2p channels, 2 p2p channels per peer
ta:963211:964404 [1] NCCL INFO Connected all trees
ta:963210:964401 [0] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 512 | 512
ta:963211:964404 [1] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 512 | 512
ta:963212:964402 [2] NCCL INFO Connected all trees
ta:963210:964401 [0] NCCL INFO 2 coll channels, 0 collnet channels, 0 nvls channels, 2 p2p channels, 2 p2p channels per peer
ta:963211:964404 [1] NCCL INFO 2 coll channels, 0 collnet channels, 0 nvls channels, 2 p2p channels, 2 p2p channels per peer
ta:963212:964402 [2] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 512 | 512
ta:963212:964402 [2] NCCL INFO 2 coll channels, 0 collnet channels, 0 nvls channels, 2 p2p channels, 2 p2p channels per peer
ta:963211:964404 [1] NCCL INFO comm 0x55f6974375f0 rank 1 nranks 4 cudaDev 1 nvmlDev 1 busId 67000 commId 0x425d648351493254 - Init COMPLETE
ta:963214:964403 [3] NCCL INFO comm 0x564925139fa0 rank 3 nranks 4 cudaDev 3 nvmlDev 3 busId c7000 commId 0x425d648351493254 - Init COMPLETE
ta:963210:964401 [0] NCCL INFO comm 0x55d838843b50 rank 0 nranks 4 cudaDev 0 nvmlDev 0 busId 25000 commId 0x425d648351493254 - Init COMPLETE
ta:963212:964402 [2] NCCL INFO comm 0x55f957686b20 rank 2 nranks 4 cudaDev 2 nvmlDev 2 busId 87000 commId 0x425d648351493254 - Init COMPLETE
[rank1]:[E402 15:37:05.656783586 ProcessGroupNCCL.cpp:607] [Rank 1] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=1, OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=600000) ran for 600005 milliseconds before timing out.
[rank1]:[E402 15:37:05.658284552 ProcessGroupNCCL.cpp:670] [Rank 1] Work WorkNCCL(SeqNum=1, OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=600000) timed out in blocking wait (TORCH_NCCL_BLOCKING_WAIT=1).
[rank2]:[E402 15:37:05.664865556 ProcessGroupNCCL.cpp:607] [Rank 2] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=1, OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=600000) ran for 600005 milliseconds before timing out.
[rank3]:[E402 15:37:05.664904305 ProcessGroupNCCL.cpp:607] [Rank 3] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=1, OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=600000) ran for 600004 milliseconds before timing out.
[rank3]:[E402 15:37:05.665997851 ProcessGroupNCCL.cpp:670] [Rank 3] Work WorkNCCL(SeqNum=1, OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=600000) timed out in blocking wait (TORCH_NCCL_BLOCKING_WAIT=1).
[rank2]:[E402 15:37:05.666002861 ProcessGroupNCCL.cpp:670] [Rank 2] Work WorkNCCL(SeqNum=1, OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=600000) timed out in blocking wait (TORCH_NCCL_BLOCKING_WAIT=1).
[rank0]:[E402 15:37:05.666784464 ProcessGroupNCCL.cpp:607] [Rank 0] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=1, OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=600000) ran for 600003 milliseconds before timing out.
[rank0]:[E402 15:37:05.668049616 ProcessGroupNCCL.cpp:670] [Rank 0] Work WorkNCCL(SeqNum=1, OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=600000) timed out in blocking wait (TORCH_NCCL_BLOCKING_WAIT=1).
ta:963214:964421 [3] NCCL INFO [Service thread] Connection closed by localRank 3
ta:963210:964427 [0] NCCL INFO [Service thread] Connection closed by localRank 0
ta:963211:964425 [1] NCCL INFO [Service thread] Connection closed by localRank 1
ta:963212:964423 [2] NCCL INFO [Service thread] Connection closed by localRank 2
ta:963212:963212 [2] NCCL INFO comm 0x55f957686b20 rank 2 nranks 4 cudaDev 2 busId 87000 - Abort COMPLETE
[rank2]:[E402 15:37:06.025771590 ProcessGroupNCCL.cpp:621] [Rank 2] Some NCCL operations have failed or timed out. Due to the asynchronous nature of CUDA kernels, subsequent GPU operations might run on corrupted/incomplete data.
[rank2]:[E402 15:37:06.025867748 ProcessGroupNCCL.cpp:627] [Rank 2] To avoid data inconsistency, we are taking the entire process down.
[rank2]:[E402 15:37:06.030361549 ProcessGroupNCCL.cpp:1664] [PG 0 (default_pg) Rank 2] Exception (either an error or timeout) detected by watchdog at work: 1, last enqueued NCCL work: 1, last completed NCCL work: -1.
[rank2]:[E402 15:37:06.030448637 ProcessGroupNCCL.cpp:1709] [PG 0 (default_pg) Rank 2] Timeout at NCCL work: 1, last enqueued NCCL work: 1, last completed NCCL work: -1.
[rank2]:[E402 15:37:06.030465226 ProcessGroupNCCL.cpp:621] [Rank 2] Some NCCL operations have failed or timed out. Due to the asynchronous nature of CUDA kernels, subsequent GPU operations might run on corrupted/incomplete data.
ta:963210:963210 [0] NCCL INFO comm 0x55d838843b50 rank 0 nranks 4 cudaDev 0 busId 25000 - Abort COMPLETE
[rank0]:[E402 15:37:06.045288927 ProcessGroupNCCL.cpp:621] [Rank 0] Some NCCL operations have failed or timed out. Due to the asynchronous nature of CUDA kernels, subsequent GPU operations might run on corrupted/incomplete data.
[rank0]:[E402 15:37:06.045340616 ProcessGroupNCCL.cpp:627] [Rank 0] To avoid data inconsistency, we are taking the entire process down.
[rank0]:[E402 15:37:06.050001793 ProcessGroupNCCL.cpp:1664] [PG 0 (default_pg) Rank 0] Exception (either an error or timeout) detected by watchdog at work: 1, last enqueued NCCL work: 1, last completed NCCL work: -1.
[rank0]:[E402 15:37:06.050097231 ProcessGroupNCCL.cpp:1709] [PG 0 (default_pg) Rank 0] Timeout at NCCL work: 1, last enqueued NCCL work: 1, last completed NCCL work: -1.
[rank0]:[E402 15:37:06.050114641 ProcessGroupNCCL.cpp:621] [Rank 0] Some NCCL operations have failed or timed out. Due to the asynchronous nature of CUDA kernels, subsequent GPU operations might run on corrupted/incomplete data.
ta:963211:963211 [1] NCCL INFO comm 0x55f6974375f0 rank 1 nranks 4 cudaDev 1 busId 67000 - Abort COMPLETE
[rank1]:[E402 15:37:06.052649104 ProcessGroupNCCL.cpp:621] [Rank 1] Some NCCL operations have failed or timed out. Due to the asynchronous nature of CUDA kernels, subsequent GPU operations might run on corrupted/incomplete data.
[rank1]:[E402 15:37:06.052688904 ProcessGroupNCCL.cpp:627] [Rank 1] To avoid data inconsistency, we are taking the entire process down.
Process Process-4:
[rank1]:[E402 15:37:06.056539448 ProcessGroupNCCL.cpp:1664] [PG 0 (default_pg) Rank 1] Exception (either an error or timeout) detected by watchdog at work: 1, last enqueued NCCL work: 1, last completed NCCL work: -1.
[rank1]:[E402 15:37:06.056633086 ProcessGroupNCCL.cpp:1709] [PG 0 (default_pg) Rank 1] Timeout at NCCL work: 1, last enqueued NCCL work: 1, last completed NCCL work: -1.
[rank1]:[E402 15:37:06.056651736 ProcessGroupNCCL.cpp:621] [Rank 1] Some NCCL operations have failed or timed out. Due to the asynchronous nature of CUDA kernels, subsequent GPU operations might run on corrupted/incomplete data.
Traceback (most recent call last):
  File "/home/dhy/anaconda3/envs/multivec/lib/python3.8/multiprocessing/process.py", line 315, in _bootstrap
    self.run()
  File "/home/dhy/anaconda3/envs/multivec/lib/python3.8/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/home/dhy/multi-vector-retrieval/script/data/../../baseline/ColBERT/colbert/infra/launcher.py", line 115, in setup_new_process
    return_val = callee(config, *args)
  File "/home/dhy/multi-vector-retrieval/script/data/../../baseline/ColBERT/colbert/indexing/collection_indexer.py", line 34, in encode
    build_index_time, encode_passage_time = encoder.run(embedding_filename, shared_lists)
  File "/home/dhy/multi-vector-retrieval/script/data/../../baseline/ColBERT/colbert/indexing/collection_indexer.py", line 66, in run
    self.setup()  # Computes and saves plan for whole collection
  File "/home/dhy/multi-vector-retrieval/script/data/../../baseline/ColBERT/colbert/indexing/collection_indexer.py", line 112, in setup
    avg_doclen_est = self._sample_embeddings(sampled_pids)
  File "/home/dhy/multi-vector-retrieval/script/data/../../baseline/ColBERT/colbert/indexing/collection_indexer.py", line 153, in _sample_embeddings
    torch.distributed.all_reduce(self.num_sample_embs)
  File "/home/dhy/anaconda3/envs/multivec/lib/python3.8/site-packages/torch/distributed/c10d_logger.py", line 79, in wrapper
    return func(*args, **kwargs)
  File "/home/dhy/anaconda3/envs/multivec/lib/python3.8/site-packages/torch/distributed/distributed_c10d.py", line 2293, in all_reduce
    work.wait()
torch.distributed.DistBackendError: [Rank 2] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=1, OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=600000) ran for 600005 milliseconds before timing out.
Process Process-2:
Traceback (most recent call last):
  File "/home/dhy/anaconda3/envs/multivec/lib/python3.8/multiprocessing/process.py", line 315, in _bootstrap
    self.run()
  File "/home/dhy/anaconda3/envs/multivec/lib/python3.8/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/home/dhy/multi-vector-retrieval/script/data/../../baseline/ColBERT/colbert/infra/launcher.py", line 115, in setup_new_process
    return_val = callee(config, *args)
  File "/home/dhy/multi-vector-retrieval/script/data/../../baseline/ColBERT/colbert/indexing/collection_indexer.py", line 34, in encode
    build_index_time, encode_passage_time = encoder.run(embedding_filename, shared_lists)
  File "/home/dhy/multi-vector-retrieval/script/data/../../baseline/ColBERT/colbert/indexing/collection_indexer.py", line 66, in run
    self.setup()  # Computes and saves plan for whole collection
  File "/home/dhy/multi-vector-retrieval/script/data/../../baseline/ColBERT/colbert/indexing/collection_indexer.py", line 112, in setup
    avg_doclen_est = self._sample_embeddings(sampled_pids)
  File "/home/dhy/multi-vector-retrieval/script/data/../../baseline/ColBERT/colbert/indexing/collection_indexer.py", line 153, in _sample_embeddings
    torch.distributed.all_reduce(self.num_sample_embs)
  File "/home/dhy/anaconda3/envs/multivec/lib/python3.8/site-packages/torch/distributed/c10d_logger.py", line 79, in wrapper
    return func(*args, **kwargs)
  File "/home/dhy/anaconda3/envs/multivec/lib/python3.8/site-packages/torch/distributed/distributed_c10d.py", line 2293, in all_reduce
    work.wait()
torch.distributed.DistBackendError: [Rank 0] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=1, OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=600000) ran for 600003 milliseconds before timing out.
Process Process-3:
Traceback (most recent call last):
  File "/home/dhy/anaconda3/envs/multivec/lib/python3.8/multiprocessing/process.py", line 315, in _bootstrap
    self.run()
  File "/home/dhy/anaconda3/envs/multivec/lib/python3.8/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/home/dhy/multi-vector-retrieval/script/data/../../baseline/ColBERT/colbert/infra/launcher.py", line 115, in setup_new_process
    return_val = callee(config, *args)
  File "/home/dhy/multi-vector-retrieval/script/data/../../baseline/ColBERT/colbert/indexing/collection_indexer.py", line 34, in encode
    build_index_time, encode_passage_time = encoder.run(embedding_filename, shared_lists)
  File "/home/dhy/multi-vector-retrieval/script/data/../../baseline/ColBERT/colbert/indexing/collection_indexer.py", line 66, in run
    self.setup()  # Computes and saves plan for whole collection
  File "/home/dhy/multi-vector-retrieval/script/data/../../baseline/ColBERT/colbert/indexing/collection_indexer.py", line 112, in setup
    avg_doclen_est = self._sample_embeddings(sampled_pids)
  File "/home/dhy/multi-vector-retrieval/script/data/../../baseline/ColBERT/colbert/indexing/collection_indexer.py", line 153, in _sample_embeddings
    torch.distributed.all_reduce(self.num_sample_embs)
  File "/home/dhy/anaconda3/envs/multivec/lib/python3.8/site-packages/torch/distributed/c10d_logger.py", line 79, in wrapper
    return func(*args, **kwargs)
  File "/home/dhy/anaconda3/envs/multivec/lib/python3.8/site-packages/torch/distributed/distributed_c10d.py", line 2293, in all_reduce
    work.wait()
torch.distributed.DistBackendError: [Rank 1] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=1, OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=600000) ran for 600005 milliseconds before timing out.
ta:963214:963214 [3] NCCL INFO comm 0x564925139fa0 rank 3 nranks 4 cudaDev 3 busId c7000 - Abort COMPLETE
[rank3]:[E402 15:37:06.255802118 ProcessGroupNCCL.cpp:621] [Rank 3] Some NCCL operations have failed or timed out. Due to the asynchronous nature of CUDA kernels, subsequent GPU operations might run on corrupted/incomplete data.
[rank3]:[E402 15:37:06.255860637 ProcessGroupNCCL.cpp:627] [Rank 3] To avoid data inconsistency, we are taking the entire process down.
[rank3]:[E402 15:37:06.258951778 ProcessGroupNCCL.cpp:1664] [PG 0 (default_pg) Rank 3] Exception (either an error or timeout) detected by watchdog at work: 1, last enqueued NCCL work: 1, last completed NCCL work: -1.
[rank3]:[E402 15:37:06.259045066 ProcessGroupNCCL.cpp:1709] [PG 0 (default_pg) Rank 3] Timeout at NCCL work: 1, last enqueued NCCL work: 1, last completed NCCL work: -1.
[rank3]:[E402 15:37:06.259055726 ProcessGroupNCCL.cpp:621] [Rank 3] Some NCCL operations have failed or timed out. Due to the asynchronous nature of CUDA kernels, subsequent GPU operations might run on corrupted/incomplete data.
Process Process-5:
Traceback (most recent call last):
  File "/home/dhy/anaconda3/envs/multivec/lib/python3.8/multiprocessing/process.py", line 315, in _bootstrap
    self.run()
  File "/home/dhy/anaconda3/envs/multivec/lib/python3.8/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/home/dhy/multi-vector-retrieval/script/data/../../baseline/ColBERT/colbert/infra/launcher.py", line 115, in setup_new_process
    return_val = callee(config, *args)
  File "/home/dhy/multi-vector-retrieval/script/data/../../baseline/ColBERT/colbert/indexing/collection_indexer.py", line 34, in encode
    build_index_time, encode_passage_time = encoder.run(embedding_filename, shared_lists)
  File "/home/dhy/multi-vector-retrieval/script/data/../../baseline/ColBERT/colbert/indexing/collection_indexer.py", line 66, in run
    self.setup()  # Computes and saves plan for whole collection
  File "/home/dhy/multi-vector-retrieval/script/data/../../baseline/ColBERT/colbert/indexing/collection_indexer.py", line 112, in setup
    avg_doclen_est = self._sample_embeddings(sampled_pids)
  File "/home/dhy/multi-vector-retrieval/script/data/../../baseline/ColBERT/colbert/indexing/collection_indexer.py", line 153, in _sample_embeddings
    torch.distributed.all_reduce(self.num_sample_embs)
  File "/home/dhy/anaconda3/envs/multivec/lib/python3.8/site-packages/torch/distributed/c10d_logger.py", line 79, in wrapper
    return func(*args, **kwargs)
  File "/home/dhy/anaconda3/envs/multivec/lib/python3.8/site-packages/torch/distributed/distributed_c10d.py", line 2293, in all_reduce
    work.wait()
torch.distributed.DistBackendError: [Rank 3] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=1, OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=600000) ran for 600004 milliseconds before timing out.
