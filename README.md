
- DP_model 是用来对比计算可识别率的，需保留
- socket.listen中是最大链接数量，按理应该判断non_initiator
- num_users 貌似作用和modelurllist一样
- 注意federated_node中，直接配置client1_ip以及dp中的local_ip，可能导致通信失败

conda env create -f environment.yml
conda install conda-pack
conda pack -n fedml -o fedml_env.tar.gz

conda init 
修改federated_node中的client1_ip为本机IP
