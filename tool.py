def print_confusion_matrix(gt, pred, class_names):
    cf_matrix = confusion_matrix(gt, pred)
    dataframe = pd.DataFrame(cf_matrix, index=class_names, columns=class_names)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(dataframe, annot=True, cbar=None,cmap="YlGnBu",fmt="d")

    plt.title("Confusion Matrix"), plt.tight_layout()

    plt.ylabel("True Class"), 
    plt.xlabel("Predicted Class")
    plt.show()
    plt.clf()

def save_confusion_matrix(gt, pred, class_names, file):
    cf_matrix = confusion_matrix(gt, pred)
    dataframe = pd.DataFrame(cf_matrix, index=class_names, columns=class_names)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(dataframe, annot=True, cbar=None,cmap="YlGnBu",fmt="d")

    plt.title("Confusion Matrix"), plt.tight_layout()

    plt.ylabel("True Class"), 
    plt.xlabel("Predicted Class")
    plt.savefig(file)
    plt.clf()