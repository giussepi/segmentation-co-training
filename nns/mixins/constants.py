# -*- coding: utf-8 -*-
""" nns/mixins/constants """


class LrShedulerTrack:
    """ Holds the options for the step logic of the learning rates scheduler """

    LOSS = 0  # for scheduler.step(val_loss)
    METRIC = 1  # for scheduler.step(val_metric)
    NO_ARGS = 2  # for scheduler.step()

    OPTIONS = (LOSS, METRIC, NO_ARGS)

    @classmethod
    def validate(cls, option):
        """
        validate the provided option is between the class defined options

        Args:
            option <int>: one of LrShedulerTrack.OPTIONS
        """
        assert option in cls.OPTIONS, f'{option} is not in {cls.__name__}.OPTIONS '

    @classmethod
    def step(cls, option, scheduler, val_metric, val_loss):
        """
        Calls the scheduler step method with the right argument based in the provided option

        Args:
            option <int>: one of LrShedulerTrack.OPTIONS
            scheduler <object>: instance of the lr scheduler
            val_metric <float>: validation metric
            val_loss <torch.Tensor> validation loss
        """
        cls.validate(option)

        if option == cls.LOSS:
            scheduler.step(val_loss)  # lr_scheduler mode min
        elif option == LrShedulerTrack.METRIC:
            scheduler.step(val_metric)  # lr_scheduler mode max (because it's using DICE)
        else:
            scheduler.step()
