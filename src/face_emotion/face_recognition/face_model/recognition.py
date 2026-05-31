from typing import Any, IO, List, Dict, Tuple, Optional, cast, Generator
import hashlib, pickle, os, time
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from numpy.typing import NDArray
from PIL import Image
#
def recognize(
	img: NDArray[Any],
	db_path: str,
	filent: bool = True
) -> List[pd.DataFrame] | List[List[Dict[str, Any]]]:
	""""""
	tic = datetime.now()
	# check db exists
	if not os.path.isdir(db_path):
		raise Exception("DB Path not found")
	file_parts = [
		"recognition",
		tic.strftime("%Y-%m-%d_%H:%M:%S")
	]
	file_name = "_".join(file_parts) + ".pkl"
	datastore_path = os.path.join(db_path, file_name)
	representations = []
	df_cols = {"identity","hash","target_x","target_y","target_w","target_h"}
	# set up datastore pickle
	if not os.path.exists(datastore_path):
		save_representations(path=datastore_path)
	representations = load_representations(path=datastore_path)
	# check for any missing keys in pickle
	for i, current_representation in enumerate(representations):
		missing_keys = df_cols - set(current_representation.keys())
		if len(missing_keys) > 0:
			raise ValueError(f"item num:{i} is missing a required key {missing_keys}, delete and remake '{datastore_path}' please")
	# load db
	storage_items = set(load_images(path=db_path))
	if len(storage_items) == 0:
		raise ValueError(f"No items found in {db_path}")
	#
	new_images, old_images, replaced_images = set(), set(), set()
	pickled_images = {representation["identity"] for representation in representations}
	new_images = storage_items - pickled_images # items added
	old_images = pickled_images - storage_items # items removed
	for representation in representations:
		identity = current_representation["identity"]
		if identity in old_images: continue
		alpha_hash = current_representation["hash"]
		beta_hash = get_image_hash(identity)
		if alpha_hash != beta_hash:
			replaced_images.add(identity)
	# update replaced images
	new_images.update(replaced_images)
	old_images.update(replaced_images)
	# remove old images
	if len(old_images) > 0:
		representation = [rep for rep in representations if rep["identity"] not in old_images]
	# find representations for new images
	if len(new_images) > 0:
		representations += find_embeddings(people=new_images)
		save_pickle = True
	#
	if save_pickle:
		save_representations(path=datastore_path, representations=representations)
	if len(representations) == 0:
		print("There are no representations; exiting the function")
		return []
	# ----------
	# representations for the facial database have been optained!
#
def save_representations(
	path: str,
	representations: Optional[List[Dict[str,Any]]] = None
) -> None:
	"""Save rep to pickle"""
	with open(path, "wb") as f:
		pickle.dump(representations or [], f, pickle.HIGHEST_PROTOCOL)
	# sign credentials? gross...
#
def load_representations(
	path: str
) -> List[Dict[str,Any]]:
	# verify credentials? gross...
	with open(path, "rb") as f:
		representations = pickle.load(f)
		if not isinstance(representations, list) or not all(
			isinstance(x, dict) for x in representations
		):
			raise ValueError("Datastore not formated properly :(")
	return cast(List[Dict[str, Any]], representations)
#
def load_images(path: str) -> Generator[str, None, None]:
	"""walks a given path and yields the images in it"""
	for r, _, f in os.walk(path):
		for file in f:
			if os.path.splitext(file)[1].lower() in {".jpg",".jpeg",".png"}:
				exact_path = os.path.join(r, file)
				with Image.open(exact_path) as img:
					if img.format.lower() in {"jpeg","png"}:
						yield exact_path
#
def get_image_hash(path: str) -> str:
	file_stats = os.stat(path)
	file_size = file_stats.st_size
	creation_time = file_stats.st_birthtime
	modification_time = file_stats.st_mtime
	properties = f"{file_size}-{creation_time}-{modification_time}"
	hasher = hashlib.sha1()
	hasher.update(properties.encode("utf-8"))
	return hasher.hexdigest()
#
def find_embeddings(
	people: Set[str],
) -> List[Dict[str,Any]]:
	"""find the embeddings of a list of images, used for loading db"""
	representations = []
	for person in people:
		file_hash = get_image_hash(person)
		try:
			img_objs: List[Dict[str, Any]] = cast(
				List[Dict[str, Any]],
				
			)
		