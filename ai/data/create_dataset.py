from datasets import load_dataset

ds = load_dataset("Andyrasika/Ecommerce_FAQ")

print(ds['train'])
# ds['train'].to_csv('ecommerce_faq_train.csv', index=False)