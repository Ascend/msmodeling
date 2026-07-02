# custom_vllm_docker_simulator.py
import subprocess
from typing import Optional, Tuple
from loguru import logger
from ms_serviceparam_optimizer.config.config import get_settings, OptimizerConfigField, VllmConfig
from ms_serviceparam_optimizer.config.custom_command import VllmCommand
from ms_serviceparam_optimizer.optimizer.interfaces.simulator import SimulatorInterface
from jinja2 import Template

class CustomVllmDockerSimulator(SimulatorInterface):
    """
    Custom VLLM simulator that manages Docker containers locally and remotely via SSH
    """
    def __init__(self, config: Optional[VllmConfig] = None, *args, **kwargs):
        if config:
            self.config = config
        else:
            settings = get_settings()
            self.config = settings.vllm
        
        # Initialize parent class with process name for residual process killing
        super().__init__(*args, process_name=self.config.process_name, **kwargs)
        
        self.local_process = None
        self.remote_process = None
        
        # Command isn't directly used since we use start_service/kill_service
        self.command = []
        
    
    @property
    def base_url(self) -> str:
        """
        Get the base URL for health checks - use local service by default
        """
        return f"http://127.0.0.1:8002/health"
    
    def update_command(self) -> None:
        """
        This method is required by the interface but not used in our implementation
        since we use custom start/kill logic
        """
        self.command = VllmCommand(self.config.command).command
        print(f"updated command: {self.command}")
    
    def before_run(self, run_params: Optional[Tuple[OptimizerConfigField]] = None):
        """
        Prepare for running the service
        """
        logger.info("Preparing to start custom VLLM Docker services...")
        super().before_run(run_params)
    
    def run(self, run_params: Optional[Tuple[OptimizerConfigField]] = None, **kwargs):
        """
        Start both local and remote VLLM services
        """
        settings = get_settings()
        new_max_batched_tokens = None
        new_max_seqs = None
        for field in settings.vllm.target_field:
            if field.name == "MAX_NUM_BATCHED_TOKENS":
                new_max_batched_tokens = field.value
            elif field.name == "MAX_NUM_SEQS":
                new_max_seqs = field.value
        print(f"new_max_batched_tokens: {new_max_batched_tokens}, new_max_seqs: {new_max_seqs}")
        # Change the jinja file directory to where you put it 
        with open("/data/linyk_test/DeepSeek-V3.2-w8a8/deploy_02_jinja.sh", "r",encoding="utf-8") as f:
            content = f.read()
        
        template = Template(content)
        rendered_content = template.render(
            max_num_batched_tokens=new_max_batched_tokens,
            max_num_seqs=new_max_seqs
        )
        
        # Change the deploy script directory to where you put it 
        with open("/data/linyk_test/DeepSeek-V3.2-w8a8/deploy_02.sh", "w",encoding="utf-8") as f:
            f.write(rendered_content)
        
        # Change the jinja file directory to where you put it 
        with open("/data/linyk_test/DeepSeek-V3.2-w8a8/deploy_03_jinja.sh", "r",encoding="utf-8") as f:
            content = f.read()

        # render vllm template
        template = Template(content)
        rendered_content = template.render(
            max_num_batched_tokens=new_max_batched_tokens,
            max_num_seqs=new_max_seqs
        )
        
        # Change the deploy script directory to where you put it 
        with open("/data/linyk_test/DeepSeek-V3.2-w8a8/deploy_03.sh", "w",encoding="utf-8") as f:
            f.write(rendered_content)
        
        '''
        change the IP address, port number, and deploy script directory according to your setup.
        IP address: change the ip from 1.1.1.1 to the ip of the remote machine you are using
        port number: change the scp port number from 11111 to the one you use (By default, scp uses port 22)
        deploy script directory: change the deploy_03.sh directory to where you actually put it
        '''
        remote_command = [
            "scp", "-P", "11111",
            "/data/linyk_test/DeepSeek-V3.2-w8a8/deploy_03.sh","1.1.1.1:/data/linyk_test/DeepSeek-V3.2-w8a8"
        ]
        
        try:
            subprocess.run(remote_command, check=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to copy deploy_03.sh to remote host. {e}")
            raise
        
        if self.process_name:
            try:
                self.kill_residual_process(self.process_name)
            except Exception as e:
                logger.error(f"Failed to kill residual process. {e}")
        
        self.before_run(run_params)
        
        # Start local service
        logger.info("Starting local VLLM service...")

        # change the docker name and deploy script according to your setup
        local_command = [
        "docker", "exec",
        "DeepSeek-V3.2-VLLM-Ascend-v0.13.0rc1",
        "bash", "-c", "source /data/linyk_test/DeepSeek-V3.2-w8a8/deploy_02.sh"
        ]
        
        # Start remote service. Please change the docker name and deploy script according to your setup
        remote_command = [
            "ssh", "1.1.1.1", "-p", "11111",
            "docker exec DeepSeek-V3.2-VLLM-Ascend-v0.13.0rc1 bash -c 'source /data/linyk_test/DeepSeek-V3.2-w8a8/deploy_03.sh'"
        ]
        
        try:
            # Start local service
            self.local_process = subprocess.Popen(
                local_command,
                stdout=None,
                stderr=None,
                text=True
            )
            self.process = self.local_process
            
            self.remote_process = subprocess.Popen(
                remote_command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True
            )
            
        except Exception as e:
            logger.error(f"Failed to start VLLM services: {e}")
            self.stop()
            raise
    
    def stop(self, del_log: bool = True):
        """
        Kill both local and remote VLLM services
        """
        local_command = "pkill -9 'VLLM'"
        remote_command = ["ssh", "1.1.1.1", "-p", "11111","pkill -9 'VLLM'"]

        # Kill local service
        print("Killing local vLLM service...")
        local_process = subprocess.Popen(
        local_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True
        )
        
        print("Killing remote vLLM service on 1.1.1.1...")
        remote_process = subprocess.Popen(
        remote_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
        )
        
        # Clean up process references
        self.local_process = None
        self.remote_process = None
        
        # Call parent stop method
        super().stop(del_log)