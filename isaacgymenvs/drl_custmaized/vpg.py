import hydra
from policy_gradient import SimplePolicyGradient


class VPG(SimplePolicyGradient):
    def __init__(self, cfg):
        self.cfg = cfg
        super().__init__(cfg)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    agent = SimplePolicyGradient(cfg)
    agent.train()


if __name__ == "__main__":
    main()
