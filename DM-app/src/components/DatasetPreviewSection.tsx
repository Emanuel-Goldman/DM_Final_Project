import { Alert, Container, Header, Spinner, Table, TextContent } from "@cloudscape-design/components";
import axios from "axios";
import { useEffect, useState } from "react";
import { API_ENDPOINT, PORT, RESOURCES } from "../consts/apiConsts";

export const DatasetPreviewSection : React.FC = () => {
    const [sample, setSample] = useState<any[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Fetches dataset sample on initial render
    useEffect(() => {
        const fetchSample = async () => {
            setError(null);
            setLoading(true);
            try {
                const res = await axios.get(`${API_ENDPOINT}:${PORT}/${RESOURCES.SAMPLE}`);
                const cleanedData = res.data.map(({ "Unnamed: 0": _, ...rest }: any) => rest); // Remove "Unnamed: 0" column
                setSample(cleanedData);
            } catch (err: any) {
                setError("Error fetching columns");
                console.error("API Error:", err);
            } 
            finally {
                setLoading(false);
            }
        };

        fetchSample();
    }, []);
    
    return (
        <div>
            <Container
                header={<Header variant="h2">Dataset description and preview</Header>}
                variant="stacked"
            >
                <TextContent>
                    <p>
                        The <a href="https://www.kaggle.com/datasets/crxxom/chess-gm-players/data">Chess All GM players Statistics 2023</a> dataset contains statistics of all Grandmaster (GM) titled players on Chess.com as of July 17, 2023. 
                        It includes details like usernames, ratings (FIDE, rapid, blitz, bullet), game history, and other relevant data. 
                    </p>
                </TextContent>
            </Container>
            {loading && <Spinner />}
            {error && <Alert type="error">{error}</Alert>}
            {!loading && !error && (
                <Table
                    variant="stacked"
                    header={
                    <Header variant="h3">Preview</Header>
                    }
                    columnDefinitions={
                        sample.length > 0
                            ? Object.keys(sample[0]).map((key) => ({
                                id: key,
                                header: key,
                                cell: (item: any) => item[key],
                            }))
                            : []
                    }
                    items={sample}
                    sortingDisabled
                    stripedRows
                    loadingText="Loading dataset preview..."
                />
            )}
        </div>
    );
}