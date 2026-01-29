                    
def run_experiment(train_dataset, device):
    epochs = 5
    batch_size = 64
    learning_rate = 0.01
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

                                                    
    model = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

                   
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
    return model


                  
