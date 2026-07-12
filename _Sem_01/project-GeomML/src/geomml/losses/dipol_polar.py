from geomml.registry import LOSSES 

@LOSSES.register("dipol_polar_loss")
def loss(self, outputs, batch):

    dipole_loss = self.criterion(
        outputs["dipole"],
        batch["dipole"],
    )

    polar_loss = self.criterion(
        outputs["polar"],
        batch["polar"],
    )

    return (
        self.dipole_weight * dipole_loss
        + self.polar_weight * polar_loss
    )