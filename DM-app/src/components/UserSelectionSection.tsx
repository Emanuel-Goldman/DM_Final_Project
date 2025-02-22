import { Alert, Container, FormField, Header, SpaceBetween, Textarea } from "@cloudscape-design/components";
import axios from "axios";
import { useEffect, useState } from "react";
import { RESOURCES, PORT, API_ENDPOINT } from "../consts/apiConsts";
import { Option } from "./types";
import { AttributeSelect, useAttributeSelection } from "./AttributeSelect";
import { AlgorithmSelect, useAlgorithmSelection } from "./AlgorithmSelect";
import { useConstraintInput } from "./ConstraintInput";

export const UserSelectionSection: React.FC = () => {
    const [columns, setColumns] = useState<Option[]>([]);
    const {
        firstAttr,
        handleFirstAttrSelect,
        secondAttr,
        handleSecondAttrSelect
    } = useAttributeSelection();
    const {
        algorithm,
        handleAlgorithmSelect
    } = useAlgorithmSelection();
    const {
        constraints,
        parsedConstraints,
        constraintError,
        parseConstraints,
        handleConstraintsChange
    } = useConstraintInput();
    
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Fetches dataset columns on initial render
    useEffect(() => {
        const fetchColumns = async () => {
            setError(null);
            try {
                const res = await axios.get(`${API_ENDPOINT}:${PORT}/${RESOURCES.COLUMNS}`);
                
                // Map columns to { label, value } format
                const columnOptions: Option[] = JSON.parse(res.data).map((col: string) => ({
                    label: col,
                    value: col,
                }));

                setColumns(columnOptions);
            } catch (err: any) {
                setError("Error fetching columns");
                console.error("API Error:", err);
            } 
        };

        if (columns.length == 0) {
            fetchColumns();
        }
    }, []);
    
    return (
        <Container
            header={
                <Header 
                    variant="h2"
                    description="Select two attributes, a ranking algorithm, and supply constraints."
                >
                    Ranking preferences
                </Header>
            }
        >
            <SpaceBetween size="s">
                <SpaceBetween size="s" direction="horizontal">
                    <AttributeSelect
                        label="Attribute 1"
                        selectedAttr={firstAttr}
                        columns={columns}
                        handleSelection={handleFirstAttrSelect}
                    />
                    <AttributeSelect
                        label="Attribute 2"
                        selectedAttr={secondAttr}
                        columns={columns}
                        handleSelection={handleSecondAttrSelect}
                    />
                    <AlgorithmSelect
                        label="Ranking algorithm"
                        selectedAlg={algorithm}
                        handleSelection={handleAlgorithmSelect}
                    />
                </SpaceBetween>
                <FormField label={"Constraints"}>
                    <Textarea
                        placeholder="Enter constraints for the weight function..."
                        value={constraints}
                        onChange={handleConstraintsChange}
                        invalid={constraintError != null}
                    />
                </FormField>
            </SpaceBetween>
        </Container>
    );
}