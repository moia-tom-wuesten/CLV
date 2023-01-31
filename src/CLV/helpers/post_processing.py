import pandas as pd
import numpy as np
import datetime
from tqdm.auto import tqdm
import torch
import matplotlib.pyplot as plt


class Postprocessing:
    def __init__(
        self,
        df: pd.DataFrame,
        model: torch,
        holdout_calender: pd.DataFrame,
        calender: pd.DataFrame,
        holdout: list,
        calibration: list,
        batch_size_pred: int,
        training_end: datetime.datetime.date,
        holdout_start: datetime.datetime.date,
    ):
        super(Postprocessing, self).__init__()
        self.df = df
        self.model = model
        self.calibration = calibration
        self.holdout = holdout
        self.batch_size_pred = batch_size_pred
        self.holdout_calender = holdout_calender
        self.calender = calender
        self.training_end = training_end
        self.holdout_start = holdout_start
        self.aggregate_prediction = pd.DataFrame({})

    # before we can start forecasting, we feed in the entire previous history
    # for each individual, to build up the cell-state memory which represents
    # each individual past history. After we feed in the last element of the
    # training sequence, the model output will be the first forecasted value
    # for the holdout period. To make things easy, we first put all calibration
    # data into a individuals*sequence_length*number_of_features shaped tensor.
    # We call this object "the seed".
    def create_calibration_sequence(self):
        """
        creates a calibration sequence of all customers
        customers*sequence*features
        output: tensor
        """
        seed_df = self.df.copy(deep=True)
        seed = np.array(
            [seed_df.values for seed_df in self.calibration], dtype=np.float32
        )
        print(
            f"The seed shape is individuals ({seed.shape[0]}) X calibration length ({seed.shape[1]}) X number of features ({seed.shape[2]})"
        )

        no_samples = seed.shape[0]
        no_timesteps = seed.shape[1]
        no_features = seed.shape[2]
        no_batches = int(np.ceil(no_samples / self.batch_size_pred))
        print(f"{no_samples=}")
        print(f"{no_timesteps=}")
        print(f"{no_features=}")
        print(f"{no_batches=}")

        # pad the last batch with 0s if needed and then
        # remove the corresponding predictions after we're done predicting
        if seed.shape[0] < (self.batch_size_pred * no_batches):
            padding = np.zeros(
                (
                    (self.batch_size_pred * no_batches) - no_samples,
                    no_timesteps,
                    no_features,
                )
            )
            print(f"{len(padding)}")
            seed = np.concatenate((seed, padding), axis=0)
        return seed, no_batches, no_samples

    def execute_prediction(self, seed: torch, no_batches: int):
        # simulate several independent scenarios
        # we take the mean to remove sampling noise
        # most improvement with 20-30 independent simulations
        # generating one scenario takes about 2mins on dual-core laptop
        # simulated scenarios can be generated in parallel
        NO_SCENARIOS = 2

        scenarios = []
        # how many time-steps does the holdout have
        holdout_length = self.holdout[0].shape[0]

        for _ in tqdm(range(NO_SCENARIOS), desc="simulating scenarios"):
            batches_predicted = []

            for j in range(no_batches):
                pred = []
                # in the beginning reset the model memory
                # model_pred.reset_hidden_state()

                # calculate batch start and end indexes
                batch_start = j * self.batch_size_pred
                batch_end = (j + 1) * self.batch_size_pred
                # batch is a dictionary which links model inputs with lists of sample features
                # batch = {}
                # batch['week'] = seed[batch_start:batch_end, :, 0:1]
                # batch['transaction'] = seed[batch_start:batch_end, :, 1:2]
                batch = torch.tensor(seed[batch_start:batch_end, :, :]).long()
                # pass the batch through the prediction model: here we discard the output
                # since we're not interested in in-sample prediction (maybe you are?), we
                # just need to build up that internal cell-state memory
                self.model.reset_cell_states(x=batch)
                prediction = self.model.predict(batch=batch)
                # prediction = model3.predict(batch)
                # we do however take the very last element of each predicted sequence:
                # as this is the first forecasted value
                pred.append(prediction[:, :, :])
                # now lets forecast all the future steps autoregressively
                for i in range(holdout_length - 1):
                    for calendar_feature in ["week"]:
                        feature = np.repeat(
                            self.holdout_calender.iloc[i][calendar_feature],
                            self.batch_size_pred,
                        )
                        feature = feature[:, np.newaxis, np.newaxis]

                    # print(pred[-1].shape)
                    batch = torch.cat(
                        (torch.tensor(feature), torch.tensor(pred[-1][:, -1:, :])),
                        dim=2,
                    )
                    # print(pred[-1][:, -1:, :])
                    # break
                    prediction = self.model.predict(batch)
                    pred.append(prediction[:, :, :])
                batches_predicted.append(pred)

            scenarios.append(batches_predicted)
        return scenarios

    def convert_predictions(self, scenarios: list, no_samples: int):
        z = []
        for scenario in scenarios:
            y = []
            for batch in scenario:
                x = []
                for time_step in batch:
                    if type(time_step) == np.ndarray:
                        x.append(time_step)
                    else:
                        complete_time_step = np.concatenate(time_step, axis=-1)
                        x.append(complete_time_step)

                y.append(np.concatenate(x, axis=1))
            # cut off the padding, if any
            z.append(np.concatenate(y, axis=0)[:no_samples, :, :])

        # predictions is a multidimensional array holding all the predicted values
        predictions = np.asarray(z)
        # the shape of the result is: (NO_SCENARIOS, no_ids, sequence_lenght, features)
        print(predictions.shape)
        return predictions

    def aggregate_predictions(self, predictions: list):
        # combine calendar with results
        self.aggregate_prediction = (
            self.calender[["year", "week"]].drop_duplicates().reset_index(drop=True)[1:]
        )

        # take the mean across multiple simulations to create the final prediction
        self.aggregate_prediction["transactions"] = np.squeeze(
            np.sum(np.mean(predictions, axis=0), axis=0)
        )
        self.aggregate_prediction = self.aggregate_prediction.reset_index()

    def actual_aggregate_data(self):
        # count aggregate stats
        aggregate_counts = self.df.copy(deep=True)
        aggregate_counts["year"] = aggregate_counts["date"].dt.year
        aggregate_counts["week"] = (aggregate_counts["date"].dt.dayofyear // 7).clip(
            upper=51
        )  # we roll the 52nd week into the 51st
        aggregate_counts = (
            aggregate_counts.groupby(["year", "week"])
            .agg({"customer_id": "count", "date": "min"})
            .reset_index()
        )
        return aggregate_counts

    def show_predictions(self, type: str):
        # plot calibration and holdout with prediction
        in_sample = self.aggregate_prediction[: -self.holdout[0].shape[0]]
        aggregate_counts = self.actual_aggregate_data()
        out_of_sample = self.aggregate_prediction[-self.holdout[0].shape[0] :]
        if type == "full":
            plt.figure(figsize=(18, 5))
            plt.plot(
                aggregate_counts.index,
                aggregate_counts["customer_id"],
                color="black",
                label="actual",
            )
            plt.plot(
                in_sample["index"],
                in_sample["transactions"],
                color="Red",
                lw=0.5,
                label="in-sample fit",
            )
            plt.plot(
                out_of_sample["index"],
                out_of_sample["transactions"],
                color="magenta",
                label="Base LSTM",
            )
            plt.axvline(
                len(aggregate_counts[aggregate_counts.date <= self.training_end]),
                linestyle=":",
            )
            plt.title("Weekly Aggregate Transactions - Actuals and Predicted")
            plt.legend()
            plt.savefig("plots_valendin/actuals_prediction.png", dpi=600)
            plt.show()
        elif type == "in_sample":
            plt.figure(figsize=(18, 5))
            plt.plot(
                aggregate_counts.index,
                aggregate_counts["customer_id"],
                color="black",
                label="actual",
            )
            plt.plot(
                in_sample["index"],
                in_sample["transactions"],
                color="Red",
                lw=0.5,
                label="in-sample fit",
            )
            plt.title("Weekly Aggregate Transactions - Actuals and Predicted")
            plt.legend()
            plt.savefig("plots_valendin/actuals_prediction.png", dpi=600)
            plt.show()
        elif type == "out_of_sample":

            plt.figure(figsize=(18, 5))
            plt.plot(
                aggregate_counts[aggregate_counts.date >= self.holdout_start].index,
                aggregate_counts[aggregate_counts.date >= self.holdout_start][
                    "customer_id"
                ],
                color="black",
                label="actual",
            )
            plt.plot(
                out_of_sample["index"],
                out_of_sample["transactions"],
                color="magenta",
                label="LSTM prediction",
            )
            plt.title("Weekly Aggregate Retail Bank Transactions - Holdout Detail")
            plt.legend()
            plt.show()
        else:
            print("No valid type.")

    def run(self):
        (
            calibration_sequence,
            no_batches,
            no_samples,
        ) = self.create_calibration_sequence()
        predictions = self.execute_prediction(calibration_sequence, no_batches)

        convertet_predictions = self.convert_predictions(predictions, no_samples)
        self.aggregate_predictions(convertet_predictions)
