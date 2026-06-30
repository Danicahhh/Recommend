import unittest

from recommender.models.mmoe import PersonalizedMMOE


class PersonalizedMMOEOptionsTest(unittest.TestCase):
    def test_gate_and_task_bias_are_configured_independently(self):
        gate_only = PersonalizedMMOE(
            input_dim=4,
            task_names=("click",),
            num_experts=2,
            expert_units=(4,),
            gate_units=(4,),
            tower_units=(4,),
            use_personalized_gate=True,
            use_task_bias=False,
        )
        self.assertTrue(gate_only.use_personalized_gate)
        self.assertFalse(gate_only.use_task_bias)

        task_bias_only = PersonalizedMMOE(
            input_dim=4,
            task_names=("click",),
            num_experts=2,
            expert_units=(4,),
            gate_units=(4,),
            tower_units=(4,),
            use_personalized_gate=False,
            use_task_bias=True,
        )
        self.assertFalse(task_bias_only.use_personalized_gate)
        self.assertTrue(task_bias_only.use_task_bias)


if __name__ == "__main__":
    unittest.main()
