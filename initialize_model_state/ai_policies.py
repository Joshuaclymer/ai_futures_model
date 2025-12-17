from classes.policies import AIPolicy
from classes.entities import Entity
from typing import list

class CurrentUSExportControls(AIPolicy):
    id = "current_us_export_controls"
    
    def __init__(self, USA : Entity, USA_allies : list[Entity], USA_rivals : list[Entity]):
        self.entities_subject_to_policy = USA_allies
        self.entities_verifying_compliance = [USA]
        self.compute_export_blacklist = USA_rivals
        self.SME_export_blacklist = USA_rivals
    
    def is_politically_viable(self) -> bool:
        return True  # Assumes current US export controls are politically viable
    
    def is_verification_capacity_sufficient(self) -> bool:
        True
    