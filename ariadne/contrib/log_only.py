# Licensed to the Technische Universität Darmstadt under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The Technische Universität Darmstadt
# licenses this file to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List

from cassis import Cas

from ariadne.classifier import Classifier
from ariadne.protocol import TrainingDocument


class LogOnlyRecommender(Classifier):
    def fit(self, documents: List[TrainingDocument], layer: str, feature: str, project_id, user_id: str):
        print(
            f"Training triggered for [{feature}] on [{layer}] in [{len(documents)}] documents from project [{project_id}] for user [{user_id}]"
        )

    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        print(
            f"Prediction triggered on document [{document_id}] for [{feature}] on [{layer}] in project [{project_id}] for user [{user_id}]"
        )
