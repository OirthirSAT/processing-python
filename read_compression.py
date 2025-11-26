import json
import os

 
locations_all = ['Aberavon', 'Aberdyfi', 'Balivanich, South Uist', 'Barra', 'Barrow-in-Furness', 'Carlisle', 'Coatham', 'Culbin', 'Dumbarton', 'Duncansby', 'Dundee', 'Great Yarmouth', 'Grimsby', 'Gwynedd', 'Islay', 'Kildonan, Arran', 'Lindisfarne', 'Magilligan Point', 'Margate', 'Near Cape Wrath', 'Norfolk', 'North Skye', 'Plymouth', 'Portavogie', 'Portsmouth', 'River Dee', 'Rochester', 'Rockcliffe', 'Rothesay', 'St Fergus', 'St Ishmael', 'Stoke, Kent', 'Stromness', 'The Wash', 'Tiree', 'Torrisdale', 'Towan Beach, Cornwall', 'Tredrissi', 'Winchelsea']

locations = ['Aberavon', 'Barra', 'Barrow-in-Furness', 'Carlisle', 'Coatham', 'Duncansby', 'Great Yarmouth', 'Grimsby', 'Gwynedd', 'Islay', 'Lindisfarne', 'Magilligan Point', 'Margate', 'Norfolk', 'Portavogie', 'Portsmouth', 'Rochester', 'Rockcliffe', 'St Ishmael', 'Stoke, Kent', 'Stromness', 'Tiree', 'Torrisdale', 'Towan Beach, Cornwall', 'Tredrissi', 'Winchelsea']
location_sizes = {loc: {} for loc in locations}

os.chdir("images")

for location in locations:
    location_sizes[location] = {
        "Raw Image": 4096*4096*4*2,
        "DEFLATE": os.path.getsize(f"{location}.tif"),
        "JPEG (Quality=100%)": os.path.getsize(f"JP100_{location}.jp2"),
        "JPEG (Quality=5%)": os.path.getsize(f"JP5_{location}.jp2"),
        "JPEG (Quality=2%)": os.path.getsize(f"JP2_{location}.jp2"),
        "Threshold PNG": os.path.getsize(f"CMP_{location}_threshold_image.png"),
        "Boundaries Chain Code": os.path.getsize(f"{location}_boundaries.npz")
    }

os.chdir("..")

with open("compression_results.json", "w") as f:
    json.dump(location_sizes, f)

import numpy as np
import matplotlib.pyplot as plt

all_values = []

# desired_keys = ["original_image", "threshold_image", "boundaries"]
desired_keys = location_sizes["Aberavon"].keys()
x_label = [key.replace("_", " ") for key in desired_keys]
for file, data in location_sizes.items():
    values = [data[key]*1e-6 for key in desired_keys if key in data]
    all_values.append(values)

all_values = np.array(all_values)
log_values = np.log(all_values)
mean_log = np.mean(log_values, axis=0)
std_log = np.std(log_values, axis=0)

mean_vals = np.exp(mean_log)
lower = np.exp(mean_log - std_log)
upper = np.exp(mean_log + std_log)


plt.figure(figsize=(12, 10))
plt.rcParams["text.usetex"] = False

colours = plt.get_cmap("tab10")
ax = plt.gca()

# for i, (file, values) in enumerate(zip(location_sizes.keys(), all_values)):
#     location = file.split("_timings",1)[0]
#     plt.plot(values, marker="o", alpha=0.5, label=location, color=colours(i % 10))

x = np.arange(len(x_label))
ax.plot(x, mean_vals, marker = "o", color='black', linewidth=2, label='Average')
ax.fill_between(x, lower, upper,color='red', alpha=0.3, label='±1 SD')

# reductions per sample
reductions_seg = 1 - (all_values[:,1] / all_values[:,0])   # segmentation reduction
reductions_vec = 1 - (all_values[:,2] / all_values[:,1])   # vectorization reduction

# mean and std of reductions
mean_seg = np.mean(reductions_seg)
std_seg  = np.std(reductions_seg)

mean_vec = np.mean(reductions_vec)
std_vec  = np.std(reductions_vec)

reductions = [mean_seg*100,mean_vec*100]
stds = [std_seg*100,std_vec*100]

print(f"Segmentation reduction: {mean_seg*100:.1f}% ± {std_seg*100:.1f}%")
print(f"Vectorization reduction: {mean_vec*100:.1f}% ± {std_vec*100:.1f}%")

for i in range(len(mean_vals) - 1):
    ax.annotate(
        f"{reductions[i]:.1f} ± {stds[i]:.1f}% reduction",
        xy=((x[i] + x[i+1]) / 2, np.sqrt(mean_vals[i] * mean_vals[i+1])),
        xytext=(60, 25),
        textcoords='offset points',
        ha='center',
        va='bottom',
        fontsize=14,
        color='darkgreen',
        arrowprops=dict(arrowstyle='->', color='darkgreen')
    )

x_label2 = ["original image", "after segmentation", "final vector output"]
plt.xticks(range(len(x_label)), x_label2, rotation=0,fontsize = 16)
plt.xlabel("Compression Stage",fontsize = 18,labelpad=20)
plt.ylabel("File Size (MB)",fontsize = 18)
plt.yticks(fontsize = 16)
plt.legend(loc = "upper right", fontsize = 16)
plt.yscale("log")

plt.grid(True, which="both", linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
