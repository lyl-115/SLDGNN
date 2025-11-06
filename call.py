import time
import os
import subprocess


def get_gpu_process_count(gpu_id=0):
    """
    使用nvidia-smi查询指定GPU上的运行进程数量
    """
    try:
        # 调用 nvidia-smi，输出GPU上运行的进程信息
        result = subprocess.run(
            ['nvidia-smi', f'--query-compute-apps=pid', f'--format=csv,noheader', f'-i', str(gpu_id)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
        pids = result.stdout.strip().split('\n')
        # 过滤空字符串
        pids = [pid for pid in pids if pid]
        return len(pids)
    except subprocess.CalledProcessError as e:
        print("查询GPU进程失败:", e)
        return 0


if __name__ == "__main__":

    name = ["Cite","Cora","Pub","Computer","Photo","Blog","CS","Physics","Wiki"] 
    # for v in [4,8,16,32]:
    for v in [16,32]:
        for i in name:
            cmd='screen -dmS shiyan{0}{1}{2} bash -c "python3 ./compute5.py -m {0} -n {1} -v {2}"'.format("SDSG",i,v)
        
            while (True):
                time.sleep(3)
                count=get_gpu_process_count(0)
                if count <1:
                    ret=os.system(cmd)
                    time.sleep(5)
                    break    
                
                elif not((i=="Physics") or (i=="CS")) and count <2:
                    ret=os.system(cmd)
                    time.sleep(5)
                    break
                
                else:
                    time.sleep(15)


            print(ret)
    # name = ["Cite","Cora","Pub","Computer","Photo","Blog","Wiki","CS","Physics"] 
    # for v in [4,8,16,32]:
    #     for i in name:
    #         cmd='screen -dmS shiyan{0}{1}{2} bash -c "python3 ./compute5.py -m {0} -n {1} -v {2}"'.format("SG",i,v)
        
    #         while (True):
    #             count=get_gpu_process_count(0)
    #             if count <1:
    #                 ret=os.system(cmd)
    #                 time.sleep(3)
    #                 break    
                
    #             elif not (((i == "Physics") or (i=="CS")) &(v==32)) & count<2:
    #                 ret=os.system(cmd)
    #                 time.sleep(3)
    #                 break
    #             else:
    #                 time.sleep(15)

    #         print(ret)

    

    # name = ["Cite","Cora","Pub","Computer","Photo","Blog","CS","Physics","Wiki"] 
    # # for v in [4,8,16,32]:
    # v=4
    # for i in name:
    #     cmd='screen -dmS shiyan{0}{1}{2} bash -c "python3 ./compute5.py -m {0} -n {1} -v {2}"'.format("supGAT",i,v)
    
    #     while (True):
    #         time.sleep(3)
    #         count=get_gpu_process_count(0)
    #         if count <1:
    #             ret=os.system(cmd)
    #             time.sleep(5)
    #             break    
            
    #         elif not((i=="Physics") or (i=="CS")) and count <2:
    #             ret=os.system(cmd)
    #             time.sleep(5)
    #             break
    #         else:
    #             time.sleep(15)


    #     print(ret)
