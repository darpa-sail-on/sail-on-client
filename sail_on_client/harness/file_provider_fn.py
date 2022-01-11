"""Set of functions used by file provider."""
from sail_on_client.errors.errors import ServerError, ProtocolError, RoundError

import csv
import datetime
import numpy as np
import nltk
import os
import json
from typing import List, Optional, Dict, Any
from sklearn.metrics.cluster import normalized_mutual_info_score
import re
import traceback


def read_gt_csv_file(file_location: str) -> List:
    """
    Read the ground truth csv file.

    Args:
        file_location: Path to the gt csv file

    Returns:
        List of entries in csv file
    """
    with open(file_location, "r") as f:
        csv_reader = csv.reader(f, delimiter=",", quotechar="|")
        return list(csv_reader)[1:]


def read_meta_data(file_location: str) -> Dict:
    """
    Read the metadata file.

    Args:
        file_location: Path to the gt csv file

    Returns:
        Content of the metadata file
    """
    with open(file_location, "r") as md:
        return json.load(md)


def get_session_info(
    folder: str, session_id: str, in_process_only: bool = True
) -> Dict[str, Any]:
    """
    Retrieve session info.

    Args:
        folder: Folder where the session file is saved
        session_id: Session id for which the info is required
        in_process_only: Flag to get information while the session is active

    Returns:
        Session information as a dict
    """
    path = os.path.join(folder, f"{str(session_id)}.json")
    if os.path.exists(path):
        with open(path, "r") as session_file:
            info = json.load(session_file)
            terminated = "termination" in info
            if terminated and in_process_only:
                raise ProtocolError(
                    "SessionEnded",
                    """The session being requested has already been terminated.
                       Please either create a new session or request a different ID""",
                )
            else:
                return info
    return {}


def get_session_test_info(folder: str, session_id: str, test_id: str) -> Dict[str, Any]:
    """
    Retrieve session info for a test.

    Args:
        folder: Folder where the session file is saved
        session_id: Session id for which the info is required
        test_id: Test id for which the info is required

    Returns:
        Session information associated with test as a dict
    """
    path = os.path.join(folder, f"{str(session_id)}.{test_id}.json")
    if os.path.exists(path):
        with open(path, "r") as session_file:
            info = json.load(session_file)
            if "completion" in info:
                raise ProtocolError(
                    "TestCompleted",
                    "The test being requested has already been completed for this session",
                )
            else:
                return info
    return {}


def write_session_log_file(structure: Dict, filepath: str) -> None:
    """
    Write session information in a log file.

    Args:
        structure: Dict with information about the session
        filepath: Path to the file where log file is saved

    Returns:
        None
    """
    with open(filepath, "w") as session_file:
        json.dump(structure, session_file, indent=2)


def log_session(
    folder: str,
    session_id: str,
    activity: str,
    test_id: Optional[str] = None,
    round_id: Optional[int] = None,
    content: Optional[Dict[str, Any]] = None,
    content_loc: Optional[str] = "round",
    return_structure: Optional[bool] = False,
) -> Optional[Dict]:
    """
    Create a log files of all session activity.

    Args:
        folder: Folder where the session file is saved
        session_id: Session id associated with the log
        test_id: Test id associated with the log
        round_id: Round id associated with the log
        content: Content of the log
        content_loc: Location where the content should be added
        return_structure: Flag to return the updated log

    Returns
        Updated log if `return_structure` is set to True
    """
    structure = get_session_info(folder, session_id)
    write_session_file = True
    if test_id is None:
        structure[activity] = {"time": [str(datetime.datetime.now())]}
        if content is not None:
            structure[activity].update(content)
    else:
        test_structure = get_session_test_info(folder, session_id, test_id)
        if activity not in test_structure:
            test_structure[activity] = {"time": [str(datetime.datetime.now())]}
        if content_loc == "activity":
            if content is not None:
                test_structure[activity].update(content)
        if round_id is not None:
            round_id_str = str(round_id)
            rounds = test_structure[activity].get("rounds", {})
            if round_id_str not in rounds:
                rounds[round_id_str] = {"time": [str(datetime.datetime.now())]}
            else:
                rounds[round_id_str]["time"].append(str(datetime.datetime.now()))
            if content_loc == "round":
                if content is not None:
                    rounds[round_id_str].update(content)
            test_structure[activity]["rounds"] = rounds
            test_structure[activity]["last round"] = round_id_str

        if not return_structure:
            write_session_log_file(
                test_structure,
                os.path.join(folder, f"{str(session_id)}.{str(test_id)}.json"),
            )

        if activity == "completion":
            session_tests = structure.get("tests", {"completed_tests": []})
            session_tests["completed_tests"].append(test_id)
            structure["tests"] = session_tests
        else:
            write_session_file = False

    if write_session_file:
        write_session_log_file(
            structure, os.path.join(folder, f"{str(session_id)}.json")
        )

    if return_structure:
        return test_structure
    return None


def read_feedback_file(
    csv_reader: "csv.reader",  # type: ignore
    feedback_ids: Optional[List[str]],
    metadata: Dict[str, Any],
    check_constrained: bool = True,
) -> Dict:
    """
    Get feedback from feedback file for the specified ids in the last submitted round.

    Args:
        csv_reader: An instance of csv reader
        feedback_ids: Element ids for which feedback is requested
        metadata: Dictionary with metadata
        check_constrained: Flag to check constraints associated with feedback

    Returns:
        Dictionary containing feedback with feedback_ids as keys
    """
    feedback_constrained = metadata.get("feedback_constrained", True)

    lines: List = list(csv_reader)

    try:
        if not check_constrained or not feedback_constrained:
            start = 0
            end = len(lines)
        else:
            # under the constrained case, we always look at the last round
            start = len(lines) - int(metadata["round_size"])
            end = start + int(metadata["round_size"])
    except KeyError:
        raise RoundError(
            "no_defined_rounds",
            "round_size not defined in metadata.",
            "".join(traceback.format_stack()),
        )

    if feedback_ids is not None:
        return {
            x[0]: list(x[1:])
            for x in [[n.strip(" \"'") for n in y] for y in lines][start:end]
            if x[0] in feedback_ids
        }
    else:
        return {
            x[0]: list(x[1:])
            for x in [[n.strip(" \"'") for n in y] for y in lines][start:end]
        }


def get_classification_feedback(
    gt_file: str,
    result_files: List[str],
    feedback_ids: List[str],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Calculate the proper feedback for classification type feedback.

    Args:
        gt_file: Path to ground truth file
        result_files: List of paths with results
        feedback_ids: Element ids for which feedback is requested
        metadata: Dictionary with metadata

    Returns:
        Dictionary containing feedback with feedback_ids as keys
    """
    if feedback_ids is None or len(feedback_ids) == 0:
        # if feedback ids not provided, limit to those in the last round
        with open(result_files[0], "r") as rf:
            result_reader = csv.reader(rf, delimiter=",")
            results = read_feedback_file(
                result_reader, None, metadata, check_constrained=True
            )
            feedback_max_ids = min(
                metadata.get("feedback_max_ids", len(results)), len(results)
            )
            feedback_ids = list(results.keys())[: int(feedback_max_ids)]

    ground_truth = read_feedback_file(
        read_gt_csv_file(gt_file),
        feedback_ids,
        metadata,
        check_constrained=feedback_ids is None or len(feedback_ids) == 0,
    )

    return {
        x: min(int(ground_truth[x][metadata["columns"][0]]), metadata["known_classes"])
        for x in ground_truth.keys()
    }


def get_classification_var_feedback(
    gt_file: str,
    result_files: List[str],
    feedback_ids: List[str],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Get feedback for video activity recognition.

    Args:
        gt_file: Path to ground truth file
        result_files: List of paths with results
        feedback_ids: Element ids for which feedback is requested
        metadata: Dictionary with metadata

    Returns:
        Dictionary containing feedback with feedback_ids as keys
    """
    ground_truth = read_feedback_file(
        read_gt_csv_file(gt_file),
        feedback_ids,
        metadata,
        check_constrained=feedback_ids is None or len(feedback_ids) == 0,
    )

    return {
        x: ground_truth[x][metadata["columns"][0] : metadata["columns"][1]]
        for x in ground_truth.keys()
    }


def get_detection_feedback(
    gt_file: str,
    result_files: List[str],
    feedback_ids: List[str],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Get detection feedback.

    Args:
        gt_file: Path to ground truth file
        result_files: List of paths with results
        feedback_ids: Element ids for which feedback is requested
        metadata: Dictionary with metadata

    Returns:
        Dictionary containing feedback with feedback_ids as keys
    """
    if feedback_ids is None or len(feedback_ids) == 0:
        # if feedback ids not provided, limit to those in the last round
        with open(result_files[0], "r") as rf:
            result_reader = csv.reader(rf, delimiter=",")
            results = read_feedback_file(
                result_reader, None, metadata, check_constrained=True
            )
            feedback_ids = list(results.keys())

    ground_truth = read_feedback_file(
        read_gt_csv_file(gt_file),
        feedback_ids,
        metadata,
        check_constrained=feedback_ids is None or len(feedback_ids) == 0,
    )

    return {x: ground_truth[x][metadata["columns"][0]] for x in ground_truth.keys()}


def get_classificaton_score_feedback(
    gt_file: str,
    result_files: List[str],
    feedback_ids: List[str],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Calculate feedback on the accuracy of classification.

    Args:
        gt_file: Path to ground truth file
        result_files: List of paths with results
        feedback_ids: Element ids for which feedback is requested
        metadata: Dictionary with metadata

    Returns:
        Dictionary containing feedback with feedback_ids as keys
    """
    ground_truth = read_feedback_file(
        read_gt_csv_file(gt_file), None, metadata, check_constrained=False
    )
    with open(result_files[0], "r") as rf:
        result_reader = csv.reader(rf, delimiter=",")
        results = read_feedback_file(
            result_reader, None, metadata, check_constrained=False
        )

    # Go through results and count number correct
    num_correct = 0
    for idx in results.keys():
        r = int(np.argmax([float(i) for i in results[idx]], axis=0))
        g = int(ground_truth[idx][metadata["columns"][0]])
        if r == g:
            num_correct += 1

    accuracy = float(num_correct) / float(len(results.keys()))
    return {"accuracy": accuracy}


def get_characterization_feedback(
    gt_file: str,
    result_files: List[str],
    feedback_ids: List[str],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Calculate the proper feedback for characterization type feedback.

    Args:
        gt_file: Path to ground truth file
        result_files: List of paths with results
        feedback_ids: Element ids for which feedback is requested
        metadata: Dictionary with metadata

    Returns:
        Dictionary containing feedback with feedback_ids as keys
    """
    # Not implemented
    raise NameError("Characterization Feedback is not supported.")
    known_classes = int(metadata["known_classes"]) + 1

    with open(result_files[0], "r") as rf:
        result_reader = csv.reader(rf, delimiter=",")
        results = read_feedback_file(result_reader, feedback_ids, metadata)
    ground_truth = read_feedback_file(
        read_gt_csv_file(gt_file), feedback_ids, metadata, check_constrained=False
    )

    # If ground truth is not novel, returns 1 is prediction is correct,
    # otherwise returns 1 if prediction is not a known class
    return {
        x: 0
        if (
            sum(ground_truth[x][1:known_classes])
            > (ground_truth[x][0] + sum(ground_truth[x][known_classes:]))
            and ground_truth[x].index(max(ground_truth[x]))
            != results[x].index(max(results[x]))
        )
        or (
            sum(ground_truth[x][1:known_classes])
            < (ground_truth[x][0] + sum(ground_truth[x][known_classes:]))
            and (results[x].index(max(results[x])) in range(1, known_classes))
        )
        else 1
        for x in ground_truth.keys()
    }


def _ensure_space(input_str: str) -> str:
    return " ".join(
        [
            x.strip()
            for x in re.split(
                r"(\W+)",
                input_str.replace(";", "")
                .replace('"', "")
                .replace("|", "")
                .replace("  ", " "),
            )
        ]
    )


def get_levenshtein_feedback(
    gt_file: str,
    result_files: List[str],
    feedback_ids: List[str],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Calculate the proper feedback for levenshtein type feedback.

    Args:
        gt_file: Path to ground truth file
        result_files: List of paths with results
        feedback_ids: Element ids for which feedback is requested
        metadata: Dictionary with metadata

    Returns:
        Dictionary containing feedback with feedback_ids as keys
    """
    ground_truth = read_feedback_file(
        read_gt_csv_file(gt_file), feedback_ids, metadata, check_constrained=False
    )
    with open(result_files[0], "r") as rf:
        result_reader = csv.reader(rf, delimiter=",")
        results = read_feedback_file(result_reader, feedback_ids, metadata)

    return {
        x: [
            nltk.edit_distance(
                _ensure_space(ground_truth[x][metadata["columns"][i]]),
                _ensure_space(results[x][0]),
            )
            for i, _ in enumerate(metadata["columns"])
        ]
        for x in results.keys()
    }


def get_cluster_feedback(
    gt_file: str,
    result_files: List[str],
    feedback_ids: List[str],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Calculate the proper feedback for cluster type feedback.

    Args:
        gt_file: Path to ground truth file
        result_files: List of paths with results
        feedback_ids: Element ids for which feedback is requested
        metadata: Dictionary with metadata

    Returns:
        Dictionary containing feedback with feedback_ids as keys
    """
    ground_truth = read_feedback_file(
        read_gt_csv_file(gt_file), feedback_ids, metadata, check_constrained=False
    )
    with open(result_files[0], "r") as rf:
        result_reader = csv.reader(rf, delimiter=",")
        results = read_feedback_file(result_reader, feedback_ids, metadata)

    if feedback_ids is None:
        feedback_ids = ground_truth.keys()

    # clear ground truth of all but relevant columns
    for key in ground_truth.keys():
        ground_truth[key] = [ground_truth[key][i] for i in metadata["columns"]]

    gt_list = []
    r_list = []
    try:
        for key in sorted(feedback_ids):
            gt_list.append([float(x) for x in ground_truth[key]])
            r_list.append([float(x) for x in results[key]])
    except Exception:
        raise ServerError(
            "MissingIds",
            """Some requested Ids are missing from either ground truth or results
            file for the current round""",
        )

    gt_np = np.array(gt_list).reshape(len(gt_list))
    r_np = np.argmax(np.array(r_list), axis=1)

    return_dict = {"nmi": normalized_mutual_info_score(gt_np, r_np)}

    for i in np.unique(r_np):
        places = np.where(r_np == i)[0]
        return_dict[str(i)] = (
            max(np.unique(gt_np[places], return_counts=True)[1]) / places.shape[0]
        )

    return return_dict


def psuedo_label_feedback(
    gt_file: str,
    feedback_ids: List[str],
    feedback_type: str,
    metadata: Dict[str, Any],
    folder: str,
    session_id: str,
) -> Dict[str, Any]:
    """
    Get psuedo label feedback for requested ids.

    Args:
        gt_file: Path to ground truth file
        feedback_ids: Element ids for which feedback is requested
        feedback_type: Type of feedback
        metadata: Dictionary with metadata
        folder: Folder where the session file is saved
        session_id: Session id for which the info is required

    Returns:
        Dictionary containing feedback with feedback_ids as keys
    """
    ground_truth = read_feedback_file(read_gt_csv_file(gt_file), feedback_ids, metadata)

    structure = get_session_info(folder, session_id)

    if "psuedo_labels" in structure:
        if feedback_type in structure["psuedo_labels"]:
            labels = structure["psuedo_labels"][feedback_type]
        else:
            structure["psuedo_labels"][feedback_type] = []
            labels = []
    else:
        structure["psuedo_labels"] = {feedback_type: []}
        labels = []

    return_dict = {}
    for x in ground_truth.keys():
        col = int(ground_truth[x][metadata["columns"][0]])
        if col not in labels:
            labels.append(col)
        return_dict[x] = labels.index(col)

    structure["psuedo_labels"][feedback_type] = labels
    write_session_log_file(structure, os.path.join(folder, f"{str(session_id)}.json"))

    return return_dict
