import psutil
import subprocess


archi = ['ResNet18', 'ResNet34', 'UNet', 'FCDenseNet', 'DnCNN_plus', 'DnCNN_minus', 'Double_DnCNN_minus', 'DnCNN_paper', 'DnCNN_paper_pT']


def kill(proc_pid):
    if psutil.pid_exists(proc_pid):
        process = psutil.Process(proc_pid)
        for proc in process.children(recursive=True):
            proc.kill()
        process.kill()


for item in archi:
    proc = subprocess.Popen(["python", "testing_slave.py", f"-a {item}"], 
                        stdout=subprocess.PIPE, 
                        shell=True)

    if "0" in str(proc.communicate()[0]):
        kill(proc.pid)
    else:
        raise Exception("Something went really wrong!")

