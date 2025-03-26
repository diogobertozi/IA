def normalize_data(data):
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)
    return normalized_data

def min_max_normalize(data):
    from sklearn.preprocessing import MinMaxScaler
    
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)
    return normalized_data

def robust_normalize(data):
    from sklearn.preprocessing import RobustScaler
    
    scaler = RobustScaler()
    normalized_data = scaler.fit_transform(data)
    return normalized_data